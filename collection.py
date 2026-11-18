"""Модуль сбора данных.

Загружает данные четырёх источников в PostgreSQL. Вызывается из main.py:
    python main.py collect                 # загрузить всё
    python main.py collect --only weather  # только один источник
    python main.py collect --full          # игнорировать инкрементальность
    python main.py collect --journal       # показать журнал загрузок

Свойства, требуемые методичкой:
  * инкрементальность — догружается только то, чего ещё нет в базе;
  * идемпотентность — повторный запуск даёт тот же результат;
  * устойчивость к сбоям — отказ одного источника не роняет остальные;
  * журнал загрузок — таблица load_log.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import yaml

from ingest import calendar_ru, cbr, rosstat, weather
from ingest.http import SourceUnavailable
from ingest.storage import Storage, utcnow

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("tourflow")


def last_complete_month_end() -> str:
    today = date.today()
    first_of_month = today.replace(day=1)
    return (first_of_month - timedelta(days=1)).isoformat()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not config["period"].get("end"):
        config["period"]["end"] = last_complete_month_end()
    return config


# --------------------------------------------------------------------------


def run_rosstat(store: Storage, config: dict, run_id: str, full: bool) -> None:
    settings = config["sources"]["rosstat"]
    path = BASE_DIR / settings["file"]
    region = config["region"]["rosstat_name"]

    store.log_start(run_id, "rosstat")
    if not path.exists():
        store.log_finish(run_id, "rosstat", "skipped",
                         message=f"файл не найден: {path}")
        log.warning("Росстат: файл %s не найден. Скачайте выгрузку "
                    "с https://rosstat.gov.ru/statistics/turizm", path)
        return

    try:
        digest = rosstat.file_hash(path)
        records = rosstat.parse(path, region, settings["indicators"])

        known = set()
        if not full:
            known = set(store.query(
                "SELECT indicator, period FROM rosstat_monthly WHERE region=%s",
                (region,),
            ))

        fresh = [r for r in records
                 if (r["indicator"], r["period"]) not in known]

        written = store.upsert(
            "rosstat_monthly",
            ["region", "indicator", "period", "value",
             "is_cumulative_source", "loaded_at"],
            [(region, r["indicator"], r["period"], r["value"], 1, utcnow())
             for r in fresh],
            key_columns=["region", "indicator", "period"],
        )
        periods = sorted(r["period"] for r in records)
        store.log_finish(run_id, "rosstat", "success",
                         rows_read=len(records), rows_written=written,
                         period_from=periods[0] if periods else None,
                         period_to=periods[-1] if periods else None,
                         message=f"sha256={digest[:16]}")
        log.info("Росстат: прочитано %s, записано %s", len(records), written)

        footnoted = {r["period"] for r in records if r["footnote"]}
        if footnoted:
            log.warning("Росстат: за периоды %s..%s подпись региона содержит "
                        "сноску — проверьте сопоставимость методики",
                        min(footnoted), max(footnoted))
    except Exception as exc:
        store.log_finish(run_id, "rosstat", "failed", message=str(exc))
        log.error("Росстат: ошибка разбора: %s", exc)


def run_weather(store: Storage, config: dict, run_id: str, full: bool) -> None:
    settings = config["sources"]["weather"]
    net = {
        "timeout": config["network"]["timeout_seconds"],
        "max_retries": config["network"]["max_retries"],
        "backoff": config["network"]["backoff_seconds"],
        "delay": config["network"]["delay_seconds"],
    }
    store.log_start(run_id, "weather")
    total_read = total_written = 0
    failures = []

    for point in config["region"]["weather_points"]:
        name = point["name"]
        start = config["period"]["start"]
        if not full:
            latest = store.max_value("weather_daily", "date",
                                     "point=%s", (name,))
            if latest:
                start = (date.fromisoformat(latest) +
                         timedelta(days=1)).isoformat()

        end = weather.effective_end(config["period"]["end"])
        if start > end:
            log.info("Погода (%s): новых дат нет", name)
            continue

        try:
            rows = weather.fetch_point(
                settings["base_url"], point["latitude"], point["longitude"],
                start, end, settings["daily"], settings["timezone"], **net,
            )
            total_read += len(rows)
            total_written += store.upsert(
                "weather_daily",
                ["point", "date", "temp_mean", "temp_max",
                 "precip_sum", "sunshine_s", "loaded_at"],
                [(name, *row, utcnow()) for row in rows],
                key_columns=["point", "date"],
            )
            log.info("Погода (%s): %s..%s, строк %s", name, start, end, len(rows))
        except SourceUnavailable as exc:
            failures.append(f"{name}: {exc}")
            log.error("Погода (%s): %s", name, exc)

    status = "failed" if failures and not total_written else "success"
    store.log_finish(run_id, "weather", status,
                     rows_read=total_read, rows_written=total_written,
                     message="; ".join(failures) or None)


def run_cbr(store: Storage, config: dict, run_id: str, full: bool) -> None:
    settings = config["sources"]["cbr"]
    net = {
        "timeout": config["network"]["timeout_seconds"],
        "max_retries": config["network"]["max_retries"],
        "backoff": config["network"]["backoff_seconds"],
        "delay": config["network"]["delay_seconds"],
    }
    store.log_start(run_id, "cbr")
    total_read = total_written = 0
    failures = []

    for currency, code in settings["currencies"].items():
        start = config["period"]["start"]
        if not full:
            latest = store.max_value("fx_daily", "date",
                                     "currency=%s", (currency,))
            if latest:
                start = (date.fromisoformat(latest) +
                         timedelta(days=1)).isoformat()

        end = config["period"]["end"]
        if start > end:
            log.info("ЦБ РФ (%s): новых дат нет", currency)
            continue

        try:
            rows = cbr.fetch_currency(settings["base_url"], code,
                                      start, end, **net)
            total_read += len(rows)
            total_written += store.upsert(
                "fx_daily",
                ["currency", "date", "rate", "loaded_at"],
                [(currency, day, rate, utcnow()) for day, rate in rows],
                key_columns=["currency", "date"],
            )
            log.info("ЦБ РФ (%s): %s..%s, строк %s",
                     currency, start, end, len(rows))
        except SourceUnavailable as exc:
            failures.append(f"{currency}: {exc}")
            log.error("ЦБ РФ (%s): %s", currency, exc)

    status = "failed" if failures and not total_written else "success"
    store.log_finish(run_id, "cbr", status,
                     rows_read=total_read, rows_written=total_written,
                     message="; ".join(failures) or None)


def run_calendar(store: Storage, config: dict, run_id: str, full: bool) -> None:
    settings = config["sources"]["calendar"]
    net = {
        "timeout": config["network"]["timeout_seconds"],
        "max_retries": config["network"]["max_retries"],
        "backoff": config["network"]["backoff_seconds"],
        "delay": config["network"]["delay_seconds"],
    }
    store.log_start(run_id, "calendar")

    first_year = int(config["period"]["start"][:4])
    # календарь известен заранее — берём и следующий год, он нужен для прогноза
    last_year = date.today().year + 1

    loaded_years = set()
    if not full:
        rows = store.query(
            "SELECT LEFT(date,4) AS y, COUNT(*) FROM calendar_daily "
            "GROUP BY y HAVING COUNT(*) >= 365"
        )
        loaded_years = {int(row[0]) for row in rows}

    total_read = total_written = 0
    failures = []

    for year in range(first_year, last_year + 1):
        # текущий и будущий год перезагружаем всегда: календарь уточняется
        if year in loaded_years and year < date.today().year:
            continue
        try:
            rows = calendar_ru.fetch_year(settings["base_url"], year,
                                          settings["country"], **net)
            total_read += len(rows)
            total_written += store.upsert(
                "calendar_daily",
                ["date", "day_type", "loaded_at"],
                [(day, kind, utcnow()) for day, kind in rows],
                key_columns=["date"],
            )
            log.info("Календарь: %s, дней %s", year, len(rows))
        except SourceUnavailable as exc:
            failures.append(f"{year}: {exc}")
            log.error("Календарь (%s): %s", year, exc)

    status = "failed" if failures and not total_written else "success"
    store.log_finish(run_id, "calendar", status,
                     rows_read=total_read, rows_written=total_written,
                     message="; ".join(failures) or None)


# --------------------------------------------------------------------------

RUNNERS = {
    "rosstat": run_rosstat,
    "weather": run_weather,
    "cbr": run_cbr,
    "calendar": run_calendar,
}


def show_journal(store: Storage, limit: int = 20) -> None:
    rows = store.query(
        "SELECT run_id, source, started_at, status, rows_read, rows_written, "
        "period_from, period_to, message FROM load_log "
        "ORDER BY started_at DESC LIMIT %s", (limit,)
    )
    if not rows:
        print("Журнал пуст.")
        return
    header = f"{'запуск':10} {'источник':10} {'время':21} {'статус':8} " \
             f"{'прочит.':>8} {'записано':>9}  период"
    print(header)
    print("-" * len(header))
    for run_id, source, started, status, read, written, pfrom, pto, msg in rows:
        period = f"{pfrom or ''}..{pto or ''}" if pfrom or pto else ""
        print(f"{run_id[:8]:10} {source:10} {started[:19]:21} {status:8} "
              f"{read:>8} {written:>9}  {period}")
        if msg:
            print(f"{'':>60}{msg[:70]}")


def collect(config: dict, only: list[str] | None = None, full: bool = False,
            journal_only: bool = False, *, run_id: str | None = None) -> int:
    """Выполнить сбор данных. Возвращает код завершения (0 — успех)."""
    try:
        store = Storage(config["storage"]["postgres"])
    except Exception as exc:
        log.error("Не удалось подключиться к PostgreSQL: %s", exc)
        log.error("Проверьте storage.postgres в config.yaml — host, port, "
                  "dbname, user, password — и что база данных создана заранее.")
        return 1
    (BASE_DIR / config["storage"]["raw_dir"]).mkdir(parents=True, exist_ok=True)

    try:
        if journal_only:
            show_journal(store)
            return 0

        run_id = run_id or uuid.uuid4().hex
        log.info("Запуск %s, период до %s", run_id[:8], config["period"]["end"])

        selected = only or list(RUNNERS)
        for name in selected:
            if not config["sources"][name]["enabled"]:
                log.info("%s: отключён в конфигурации", name)
                continue
            RUNNERS[name](store, config, run_id, full)

        print()
        show_journal(store, limit=len(selected))
        return 0
    finally:
        store.close()
