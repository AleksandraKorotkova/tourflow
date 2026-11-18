"""Фоновый запуск этапов TourFlow и чтение журнала PostgreSQL."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import sys
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".tourflow"
CONFIG = ROOT / "config.yaml"
MART = ROOT / "data" / "monthly_from_db.csv"
SAMPLE_MART = ROOT / "data" / "monthly.csv"
SAMPLE_RESULTS = ROOT / "output_from_db"
RUNS = STATE / "runs"
SOURCES = {
    "rosstat": "Росстат", "weather": "Погода", "cbr": "Банк России",
    "calendar": "Календарь",
}


def active_results() -> Path:
    completed = sorted(path for path in RUNS.glob("*/run.json")
                       if not path.parent.name.startswith("."))
    return completed[-1].parent if completed else SAMPLE_RESULTS


def offline_mart() -> Path:
    return MART if MART.is_file() else SAMPLE_MART


def read_config() -> dict:
    from main import load_config
    return load_config(CONFIG)


def rosstat_path() -> Path:
    # Для выбора файла достаточно прочитать настройки, без импорта моделей.
    import yaml

    file = None
    for path in (CONFIG, ROOT / "config.local.yaml"):
        if path.is_file():
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            file = config.get("sources", {}).get("rosstat", {}).get("file", file)
    return ROOT / (file or "data/raw/SR_mes_sub.xlsx")


def check_excel(path: Path) -> None:
    if path.suffix.lower() not in {".xlsx", ".xlsm"} or not path.is_file():
        raise ValueError("Выберите существующий файл Росстата формата .xlsx или .xlsm.")


def journal(run_id: str | None = None, *, pg_config: dict | None = None) -> list[tuple]:
    import psycopg2
    from psycopg2.errors import UndefinedTable

    if pg_config is None:
        pg_config = read_config()["storage"]["postgres"]
    settings = {"connect_timeout": 5, **pg_config}
    try:
        with closing(psycopg2.connect(**settings)) as connection:
            connection.set_session(readonly=True)
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = '5000ms'")
                query = (
                    "SELECT run_id, source, started_at, status, rows_read, rows_written, "
                    "period_from, period_to, message FROM public.load_log"
                )
                if run_id is None:
                    cursor.execute(query + " ORDER BY started_at DESC, source LIMIT 100")
                else:
                    cursor.execute(query + " WHERE run_id = %s ORDER BY source", (run_id,))
                return cursor.fetchall()
    except UndefinedTable:
        if run_id is not None:
            raise
        return []


@contextmanager
def calculation_lock():
    STATE.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(STATE / "calculation.db", timeout=0)) as connection:
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                raise RuntimeError("Расчёт уже выполняется в другом окне TourFlow.") from exc
            raise
        yield


def phase(title: str) -> None:
    print(f"@@PHASE {title}", flush=True)


def collect_sources(config: dict, excel: Path) -> list[str]:
    from collection import collect

    if config["sources"]["rosstat"]["enabled"]:
        check_excel(excel)
        config["sources"]["rosstat"]["file"] = str(excel)
    # Таймаут действует и для подключения на этапе сборки витрины.
    config["storage"]["postgres"].setdefault("connect_timeout", 5)
    phase("1 из 3 · Загрузка источников")
    run_id = uuid.uuid4().hex
    if collect(config, run_id=run_id):
        raise RuntimeError("Загрузка не запустилась. Подробности в ходе выполнения.")

    warnings = []
    rows = journal(run_id, pg_config=config["storage"]["postgres"])
    for _, source, _, status, read, written, _, _, message in rows:
        name = SOURCES.get(source, source)
        if source == "rosstat":
            if status != "success":
                raise RuntimeError(f"Росстат: {message or 'загрузка не завершена'}")
            if not read:
                raise RuntimeError("В Excel не найдены показатели Росстата для выбранного региона.")
        if status != "success" or (source != "rosstat" and message):
            warnings.append(name)
            print(f"Предупреждение · {name}: {message or status}")
        elif not written:
            print(f"{name}: новых записей нет.")
    if warnings:
        print("Для сборки витрины будут использованы доступные записи из PostgreSQL.")
    return warnings


def run(action: str, excel: Path | None = None) -> None:
    import main

    with calculation_lock():
        config = read_config()
        warnings = []
        if action == "update":
            if excel is None:
                raise ValueError("Выберите файл Росстата.")
            warnings = collect_sources(config, excel)

        RUNS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        name = f"{stamp}-{uuid.uuid4().hex[:8]}"
        staging = RUNS / f".{name}"
        staging.mkdir()
        try:
            if action == "update":
                phase("2 из 3 · Сборка месячной витрины")
                built_mart = staging / "monthly_from_db.csv"
                if main.step_build(config, built_mart):
                    raise RuntimeError("Не удалось собрать месячную витрину.")
                phase("3 из 3 · Анализ и прогноз")
            else:
                phase("Анализ подготовленных данных")

            mart = built_mart if action == "update" else offline_mart()
            if not mart.is_file():
                raise FileNotFoundError(f"Файл витрины не найден: {mart}")
            metadata = {}
            if main.step_analyze(config, mart, staging, None, summary=metadata):
                raise RuntimeError("Анализ завершился с ошибкой.")
            metadata.update({
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "warnings": warnings,
            })
            (staging / "run.json").write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )
            if action == "update":
                MART.parent.mkdir(parents=True, exist_ok=True)
                built_mart.replace(MART)
            completed = staging.rename(RUNS / name)
            print(f"Результаты: {completed}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        action = sys.argv[1]
        if action not in {"analyze", "update"}:
            raise ValueError("Неизвестная команда.")
        run(action, Path(sys.argv[2]) if action == "update" else None)
    except Exception as exc:
        logging.exception("Не удалось завершить операцию: %s", exc)
        sys.exit(1)
