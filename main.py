#!/usr/bin/env python3
"""Главный файл прогнозно-аналитической системы.

Сам ничего не считает: читает конфигурацию, вызывает функции модулей
и печатает итог.

Запуск:
    python main.py collect            # собрать данные из источников в PostgreSQL
    python main.py build              # собрать месячную витрину из PostgreSQL в CSV
    python main.py analyze            # анализ, прогноз и графики (по умолчанию)
    python main.py all                # collect + build + analyze

Параметры:
    --config config.yaml   файл конфигурации
    --csv PATH             другой файл витрины
    --out DIR              другая папка для результатов
    --horizon N            горизонт прогноза, месяцев
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

import analysis
import forecasting
import processing
import visualization

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("tourflow")


def load_config(path: Path) -> dict:
    """Прочитать конфигурацию; локальные настройки (пароли) из
    config.local.yaml, если файл есть, перекрывают основные."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    local = path.with_name("config.local.yaml")
    if local.exists():
        with open(local, encoding="utf-8") as handle:
            _merge(config, yaml.safe_load(handle) or {})
    if not config["period"].get("end"):
        from datetime import date, timedelta
        config["period"]["end"] = (date.today().replace(day=1)
                                   - timedelta(days=1)).isoformat()
    return config


def _merge(base: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# Сценарии
# ---------------------------------------------------------------------------

def step_collect(config: dict) -> int:
    import collection  # отдельно: требует psycopg2
    return collection.collect(config)


def step_build(config: dict, csv_path: Path) -> int:
    """Собрать витрину и сохранить в PostgreSQL и CSV."""
    pg_config = config["storage"]["postgres"]

    frame = processing.build_mart_from_db(pg_config)
    saved = processing.save_mart_to_db(frame, pg_config)
    processing.save_mart(frame, csv_path)

    print(f"Витрина в PostgreSQL: public.monthly_mart, строк: {saved}")
    print(f"Витрина в CSV: {csv_path}")
    print(
        f"Периодов: {len(frame)} "
        f"({frame.index.min()} .. {frame.index.max()}), "
        f"столбцов: {len(frame.columns)}"
    )
    return 0


def step_analyze(config: dict, csv_path: Path, out_dir: Path,
                 horizon: int | None) -> int:
    """Обработка -> анализ -> прогнозирование -> визуализация -> выводы."""
    settings = config["analysis"]
    target = settings["target"]
    label = settings["target_label"]
    fsettings = forecasting.ForecastSettings.from_dict(settings.get("forecast"))
    if horizon:
        fsettings.horizon = horizon

    # 1. обработка
    raw_frame = processing.load_mart(csv_path)
    processing.target_series(raw_frame, target)
    frame, quality = processing.prepare(raw_frame, target)
    series = processing.target_series(frame, target)
    print(f"Данные: {quality['observed']} месяцев показателя "
          f"({quality['period'][0]} .. {quality['period'][1]}), "
          f"пропусков внутри ряда: {len(quality['gaps'])}")
    if quality["calendar_bad"]:
        print(
            f"Внимание: календарь недостоверен для лет "
            f"{quality['calendar_bad']}. В прогнозных моделях для этих лет "
            f"используется допущение: число нерабочих дней соответствует "
            f"норме 2010–2021 годов. Это не фактический календарь."
        )

    # 2. анализ
    season = analysis.fit_seasonal_trend(series)
    anomalies = analysis.detect_anomalies(season["residuals"],
                                          settings["anomaly_threshold"])
    correlations = analysis.factor_correlations(season["residuals"], frame,
                                                settings["factors"])
    lags = (analysis.lag_correlations(season["residuals"], frame["fx_usd_yoy"])
            if "fx_usd_yoy" in frame.columns else None)

    # 3. прогнозирование
    result = forecasting.run_forecasting(frame, series, fsettings)
    best_forecast = result.forecasts[result.best]

    # 4. выводы
    horizon_label = f"{fsettings.horizon} мес."
    conclusions = analysis.make_conclusions(
        series, season, anomalies, correlations, result.metrics,
        best_forecast, result.best, horizon_label)

    # 5. сохранение результатов
    out_dir.mkdir(parents=True, exist_ok=True)
    interval_checks = forecasting.check_interval_coverage(
        result.backtest, fsettings.interval_level
    )
    interval_checks.to_csv(
        out_dir / "interval_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not interval_checks.empty:
        interval_summary = (
            interval_checks.groupby(["model", "h"])
            .agg(
                coverage=("covered", "mean"),
                n=("covered", "size"),
                mean_width=("width", "mean"),
            )
        )
        interval_summary["coverage_pct"] = (
            interval_summary.pop("coverage") * 100
        )
        interval_summary.to_csv(
            out_dir / "interval_coverage.csv",
            encoding="utf-8-sig",
        )
        print(
            "\nИсторическое покрытие интервалов "
            f"(заданный уровень {fsettings.interval_level:.0%}):"
        )
        print(interval_summary.round(1).to_string())
        print(
            "n — число проверенных интервалов; "
            "mean_width — средняя ширина, человек."
        )
    else:
        print(
            "\nНедостаточно прошлых ошибок для проверки "
            "покрытия интервалов."
        )
    result.backtest.to_csv(
        out_dir / "backtest.csv", index=False, encoding="utf-8-sig"
    )
    result.by_horizon.to_csv(
        out_dir / "metrics_by_horizon.csv", encoding="utf-8-sig"
    )
    result.effects.to_csv(
        out_dir / "regression_effects.csv", encoding="utf-8-sig"
    )
    frame.to_csv(
        out_dir / "prepared_monthly.csv", encoding="utf-8-sig"
    )

    import json
    (out_dir / "quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.metrics.round(2).to_csv(out_dir / "metrics.csv", encoding="utf-8-sig")
    best_forecast.round(0).rename_axis("period").to_csv(out_dir / "forecast.csv",
                                                       encoding="utf-8-sig")
    anomalies.round(3).to_csv(out_dir / "anomalies.csv", encoding="utf-8-sig")
    correlations.round(3).to_csv(out_dir / "correlations.csv", encoding="utf-8-sig")
    (out_dir / "conclusions.txt").write_text("\n\n".join(conclusions), encoding="utf-8")
    figures = visualization.build_all(
        out_dir / "figures", series, season, anomalies,
        lags if lags is not None else analysis.pd.DataFrame(),
        result, fsettings.interval_level, label)

    # 6. сводка в консоль
    print("\nТочность моделей (скользящая проверка):")
    print(result.metrics.round(1).to_string())
    print(
        "\nMAPE — средняя абсолютная процентная ошибка прогноза."
        "\nRMSE — среднеквадратичная ошибка, человек."
        "\nn — число пар «факт–прогноз»; проверочные периоды перекрываются."
        "\nЛучшая модель выбрана по средней MAPE для всех шагов прогноза."
    )
    for name in result.skipped:
        print(f"  пропущена: {name} (установите statsmodels)")
    print(f"\nЛучшая модель: {result.best}")
    print("\nПрогноз:")
    shown = best_forecast.copy()
    shown.index = [analysis.fmt_period(p) for p in shown.index]
    print(shown.round(0).map(analysis.fmt_int).to_string())
    print("\nВыводы:")
    for line in conclusions:
        print(f"  • {line}")
    print(f"\nРезультаты: {out_dir}  (графиков: {len(figures)})")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="analyze",
                        choices=["collect", "build", "analyze", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--csv", help="файл витрины (по умолчанию из конфигурации)")
    parser.add_argument("--out", help="папка результатов")
    parser.add_argument("--horizon", type=int, help="горизонт прогноза, месяцев")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    config = load_config(BASE_DIR / args.config)
    csv_path = BASE_DIR / (args.csv or config["analysis"]["mart_csv"])
    out_dir = BASE_DIR / (args.out or config["analysis"]["output_dir"])

    if args.command in ("collect", "all"):
        code = step_collect(config)
        if code:
            return code
    if args.command in ("build", "all"):
        code = step_build(config, csv_path)
        if code:
            return code
    if args.command in ("analyze", "all"):
        return step_analyze(config, csv_path, out_dir, args.horizon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
