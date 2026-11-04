"""Модуль обработки данных.

Отвечает за всё, что происходит с данными до анализа:
  * получение месячной витрины — из CSV-файла или из PostgreSQL;
  * контроль качества (пропуски, отрицательные значения, правдоподобие
    производственного календаря);
  * расчёт производных показателей;
  * расчёт признаков для анализа и прогнозирования: отклонение числа
    нерабочих дней от типичного для месяца и годовое изменение курса.

Все вычисления оформлены функциями; модуль ничего не печатает и не рисует,
результат возвращается значениями. Индекс витрины — pandas.PeriodIndex
с месячной частотой.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
def read_query(conn, sql: str, parse_dates=()) -> pd.DataFrame:
    """Прочитать SQL-запрос через курсор PostgreSQL."""
    with conn.cursor() as cursor:
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description]
        frame = pd.DataFrame(cursor.fetchall(), columns=columns)

    for column in parse_dates:
        frame[column] = pd.to_datetime(frame[column])

    return frame

# Календарный год, в котором доля нерабочих дней ниже порога, считается
# недостоверным: в нормальном году нерабочих дней около 32 % (11–12 из 36).
MIN_PLAUSIBLE_NONWORKING_SHARE = 0.20


# ---------------------------------------------------------------------------
# Загрузка и сохранение витрины
# ---------------------------------------------------------------------------

def load_mart(csv_path: str | Path) -> pd.DataFrame:
    """Прочитать месячную витрину из CSV и привести индекс к PeriodIndex."""
    frame = pd.read_csv(csv_path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    frame.index.name = "period"
    return frame.sort_index()


def save_mart(frame: pd.DataFrame, csv_path: str | Path) -> Path:
    """Сохранить витрину в CSV (UTF-8 с BOM — корректно открывается в Excel)."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.index = out.index.astype(str)
    out.to_csv(path, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------------------
# Сборка витрины из PostgreSQL
# ---------------------------------------------------------------------------

def read_rosstat(conn) -> pd.DataFrame:
    """Показатели Росстата: из длинного формата в широкий (колонка = показатель)."""
    frame = read_query(
        conn,
        "SELECT period, indicator, value FROM rosstat_monthly",
    )
    if frame.empty:
        return pd.DataFrame()
    wide = frame.pivot(index="period", columns="indicator", values="value")
    wide.columns.name = None
    return wide


def read_weather(conn, norm_from: int, norm_to: int) -> pd.DataFrame:
    """Месячная погода и отклонения от нормы.

    Используются только месяцы с данными за каждый день
    и без пропусков в погодных показателях.
    """
    daily = read_query(
        conn,
        "SELECT point, date, temp_mean, temp_max, precip_sum, sunshine_s "
        "FROM weather_daily",
        parse_dates=["date"],
    )
    if daily.empty:
        return pd.DataFrame()

    weather_columns = [
        "temp_mean", "temp_max", "precip_sum", "sunshine_s"
    ]
    daily["period"] = daily["date"].dt.strftime("%Y-%m")

    grouped = daily.groupby(["point", "period"])
    completeness = grouped[weather_columns].count()
    completeness["unique_days"] = grouped["date"].nunique()
    completeness["rows"] = grouped.size()
    completeness["expected_days"] = [
        pd.Period(period, freq="M").days_in_month
        for _, period in completeness.index
    ]

    valid = completeness[
        weather_columns + ["unique_days", "rows"]
    ].eq(completeness["expected_days"], axis=0).all(axis=1)

    excluded = completeness.index[~valid]
    if len(excluded):
        log.warning(
            "Неполные погодные месяцы исключены из витрины: %s",
            ", ".join(
                f"{point}: {period}"
                for point, period in excluded
            ),
        )

    valid_keys = completeness.index[valid]
    keys = pd.MultiIndex.from_frame(daily[["point", "period"]])
    daily = daily.loc[keys.isin(valid_keys)].copy()

    if daily.empty:
        return pd.DataFrame()

    daily["month"] = daily["date"].dt.month
    daily["is_warm"] = (daily["temp_max"] >= 25).astype(int)
    daily["is_rainy"] = (daily["precip_sum"] >= 1.0).astype(int)

    monthly = daily.groupby(["point", "period", "month"]).agg(
        temp_mean=("temp_mean", "mean"),
        warm_days=("is_warm", "sum"),
        rainy_days=("is_rainy", "sum"),
        precip_sum=("precip_sum", "sum"),
        sunshine_h=("sunshine_s", lambda s: s.sum() / 3600.0),
    ).reset_index()

    monthly["year"] = monthly["period"].str[:4].astype(int)
    reference = monthly[
        monthly["year"].between(norm_from, norm_to)
    ]

    metrics = [
        "temp_mean", "warm_days", "rainy_days",
        "precip_sum", "sunshine_h"
    ]
    norm = reference.groupby(["point", "month"])[metrics].mean()
    norm.columns = [f"{column}_norm" for column in metrics]

    merged = monthly.merge(norm, on=["point", "month"], how="left")
    for metric in metrics:
        merged[f"{metric}_anom"] = (
            merged[metric] - merged[f"{metric}_norm"]
        )

    keep = (
        ["period", "point"]
        + metrics
        + [f"{metric}_anom" for metric in metrics]
    )
    wide = merged[keep].pivot(index="period", columns="point")
    wide.columns = [
        f"{point}_{metric}" for metric, point in wide.columns
    ]
    return wide.sort_index()


def read_fx(conn) -> pd.DataFrame:
    """Среднемесячные курсы валют."""
    daily = read_query(
        conn,
        "SELECT currency, date, rate FROM fx_daily",
        parse_dates=["date"],
    )
    if daily.empty:
        return pd.DataFrame()
    daily["period"] = daily["date"].dt.strftime("%Y-%m")
    monthly = daily.groupby(["period", "currency"])["rate"].mean().unstack()
    monthly.columns = [f"fx_{c.lower()}" for c in monthly.columns]
    return monthly


def read_calendar(conn) -> pd.DataFrame:
    """Число нерабочих и сокращённых дней в месяце."""
    daily = read_query(
        conn,
        "SELECT date, day_type FROM calendar_daily",
        parse_dates=["date"],
    )
    if daily.empty:
        return pd.DataFrame()
    daily["period"] = daily["date"].dt.strftime("%Y-%m")
    grouped = daily.groupby("period").agg(
        days_in_month=("day_type", "size"),
        nonworking_days=("day_type", lambda s: int((s == 1).sum())),
        short_days=("day_type", lambda s: int((s == 2).sum())),
    )
    grouped["nonworking_share"] = grouped["nonworking_days"] / grouped["days_in_month"]
    return grouped


def build_mart_from_db(pg_config: dict, norm_years: tuple[int, int] = (2010, 2021)
                       ) -> pd.DataFrame:
    """Собрать месячную витрину из PostgreSQL (все четыре источника)."""
    import psycopg2  # импорт здесь: для работы с CSV драйвер не нужен

    conn = psycopg2.connect(**pg_config)
    try:
        parts = [
            read_rosstat(conn),
            read_weather(conn, *norm_years),
            read_fx(conn),
            read_calendar(conn),
        ]
    finally:
        conn.close()

    parts = [p for p in parts if not p.empty]
    if not parts:
        raise RuntimeError("В базе нет данных. Сначала выполните сбор: python main.py collect")

    frame = parts[0]
    for part in parts[1:]:
        frame = frame.join(part, how="outer")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    frame.index.name = "period"
    return add_derived(frame.sort_index()).round(4)


# ---------------------------------------------------------------------------
# Производные показатели
# ---------------------------------------------------------------------------

def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать показатели, которые Росстат не публикует напрямую."""
    frame = frame.copy()
    if {"nights_total", "placed_total"} <= set(frame.columns):
        frame["avg_stay_nights"] = frame["nights_total"] / frame["placed_total"]
    if {"revenue_total", "nights_total"} <= set(frame.columns):
        # доходы в выгрузке Росстата — в тысячах рублей
        frame["revenue_per_night"] = frame["revenue_total"] * 1000 / frame["nights_total"]
    if {"placed_sanatorium", "placed_total"} <= set(frame.columns):
        frame["sanatorium_share"] = frame["placed_sanatorium"] / frame["placed_total"]
        frame["placed_non_sanatorium"] = frame["placed_total"] - frame["placed_sanatorium"]
    if {"placed_foreign", "placed_total"} <= set(frame.columns):
        frame["foreign_share"] = frame["placed_foreign"] / frame["placed_total"]
    return frame


# ---------------------------------------------------------------------------
# Контроль качества
# ---------------------------------------------------------------------------

def unreliable_calendar_years(frame: pd.DataFrame,
                              min_share: float = MIN_PLAUSIBLE_NONWORKING_SHARE
                              ) -> list[int]:
    """Годы, для которых производственный календарь недостоверен.

    Сервис isdayoff.ru для ещё не утверждённого года возвращает строку из
    одних нулей: длина верна, символы допустимы, но «года без выходных» не
    бывает. Такой ответ проходит формальную проверку загрузчика, поэтому
    правдоподобие проверяется здесь — по доле нерабочих дней за год.
    """
    if not {"nonworking_days", "days_in_month"} <= set(frame.columns):
        return []
    by_year = frame.groupby(frame.index.year)[["nonworking_days", "days_in_month"]].sum()
    share = by_year["nonworking_days"] / by_year["days_in_month"]
    return [int(year) for year, value in share.items() if value < min_share]


def check_quality(frame: pd.DataFrame, target: str) -> dict:
    """Сводный отчёт о качестве данных.

    Возвращает словарь:
      observed       — число месяцев с наблюдаемым значением показателя;
      period         — границы ряда показателя;
      gaps           — пропущенные месяцы внутри ряда показателя;
      negatives      — столбцы Росстата с отрицательными значениями;
      calendar_bad   — годы с недостоверным календарём;
      coverage       — доля заполненных значений по каждому столбцу.
    """
    series = frame[target].dropna()
    full = pd.period_range(series.index.min(), series.index.max(), freq="M")
    gaps = [str(p) for p in full.difference(series.index)]

    rosstat_cols = [c for c in frame.columns
                    if c.startswith(("placed_", "nights_", "revenue_"))]
    negatives = [c for c in rosstat_cols if (frame[c].dropna() < 0).any()]

    return {
        "observed": int(series.size),
        "period": (str(series.index.min()), str(series.index.max())),
        "gaps": gaps,
        "negatives": negatives,
        "calendar_bad": unreliable_calendar_years(frame),
        "coverage": frame.notna().mean().round(3).to_dict(),
    }


# ---------------------------------------------------------------------------
# Признаки для анализа и прогнозирования
# ---------------------------------------------------------------------------

def add_calendar_deviation(frame: pd.DataFrame) -> pd.DataFrame:
    """Отклонение числа нерабочих дней от типичного для календарного месяца.

    Сырое число нерабочих дней на 83 % определяется номером месяца и потому
    дублирует сезонность. Вычитание типичного значения месяца (медианы по
    достоверным годам) оставляет только нетипичную часть — длинные праздники,
    переносы выходных. Для недостоверных лет значение остаётся пустым.
    """
    frame = frame.copy()
    if "nonworking_days" not in frame.columns:
        return frame

    bad_years = unreliable_calendar_years(frame)
    reference_mask = (
            (frame.index.year >= 2010)
            & (frame.index.year <= 2021)
            & ~frame.index.year.isin(bad_years)
    )
    reference = frame.loc[reference_mask]

    counts = reference.groupby(reference.index.month)["nonworking_days"].count()
    if len(counts) != 12 or (counts < 3).any():
        raise ValueError(
            "Недостаточно календарных данных за 2010–2021 годы "
            "для расчёта нормы каждого месяца."
        )

    typical = reference.groupby(reference.index.month)["nonworking_days"].median()

    deviation = frame["nonworking_days"] - frame.index.month.map(typical).to_numpy()
    deviation[frame.index.year.isin(bad_years)] = np.nan
    frame["nonworking_dev"] = deviation
    if bad_years:
        log.warning("Календарь на %s недостоверен (доля нерабочих дней < %.0f %%): "
                    "отклонение для этих лет не рассчитывается",
                    bad_years, MIN_PLAUSIBLE_NONWORKING_SHARE * 100)
    return frame


def add_fx_change(frame: pd.DataFrame, columns: tuple[str, ...] = ("fx_usd",)
                  ) -> pd.DataFrame:
    """Годовое изменение курса: значение месяца к тому же месяцу год назад.

    Уровень курса нестационарен; относительное изменение за год сопоставимо
    между периодами и не содержит сезонной составляющей.
    """
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[f"{col}_yoy"] = frame[col] / frame[col].shift(12) - 1
    return frame


def prepare(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """Полная подготовка: проверка качества + производные признаки.

    Возвращает (подготовленная витрина, отчёт о качестве).
    """
    report = check_quality(frame, target)
    prepared = add_fx_change(add_calendar_deviation(add_derived(frame)))
    return prepared, report


def target_series(frame: pd.DataFrame, target: str) -> pd.Series:
    """Получить непрерывный положительный месячный ряд для моделей."""
    if target not in frame.columns:
        raise ValueError(f"В данных нет показателя: {target}")

    if frame.index.has_duplicates:
        raise ValueError("В таблице обнаружены повторяющиеся месяцы.")

    series = frame[target].dropna().astype(float).sort_index()
    series.name = target

    if series.empty:
        raise ValueError(f"Показатель {target} не содержит данных.")

    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("В основном показателе есть бесконечные значения.")

    if (series <= 0).any():
        periods = ", ".join(series.index[series <= 0].astype(str))
        raise ValueError(
            f"Для текущих моделей показатель должен быть положительным. "
            f"Проверьте месяцы: {periods}"
        )

    expected = pd.period_range(
        series.index.min(), series.index.max(), freq="M"
    )
    missing = expected.difference(series.index)
    if len(missing):
        raise ValueError(
            "Внутри ряда пропущены месяцы: "
            + ", ".join(missing.astype(str))
            + ". Перед прогнозированием необходимо обработать пропуски."
        )

    return series


def save_mart_to_db(frame: pd.DataFrame, pg_config: dict) -> int:
    """Сохранить текущую месячную витрину в public.monthly_mart.

    Замена содержимого выполняется в одной транзакции.
    Исходные таблицы источников не изменяются.
    """
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import execute_values

    if frame.empty:
        raise ValueError("Нельзя сохранить пустую витрину.")

    if frame.index.has_duplicates:
        raise ValueError("В витрине есть повторяющиеся месяцы.")

    columns = list(frame.columns)
    if len(columns) != len(set(columns)):
        raise ValueError("В витрине есть повторяющиеся столбцы.")

    if "period" in columns:
        raise ValueError("period должен находиться в индексе витрины.")

    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("В витрине обнаружены бесконечные значения.")

    rows = [
        (
            str(period),
            *[
                None if pd.isna(value) else float(value)
                for value in row
            ],
        )
        for period, row in zip(frame.index, values)
    ]

    conn = psycopg2.connect(**pg_config)
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.monthly_mart (
                        period DATE PRIMARY KEY
                    )
                    """
                )

                for column in columns:
                    cursor.execute(
                        sql.SQL(
                            "ALTER TABLE public.monthly_mart "
                            "ADD COLUMN IF NOT EXISTS {} DOUBLE PRECISION"
                        ).format(sql.Identifier(column))
                    )

                cursor.execute("DELETE FROM public.monthly_mart")

                query = sql.SQL(
                    "INSERT INTO public.monthly_mart ({}) VALUES %s"
                ).format(
                    sql.SQL(", ").join(
                        sql.Identifier(column)
                        for column in ["period"] + columns
                    )
                )

                # В БД месяц хранится датой первого числа.
                db_rows = [
                    (f"{row[0]}-01", *row[1:])
                    for row in rows
                ]
                execute_values(
                    cursor,
                    query.as_string(conn),
                    db_rows,
                )
    finally:
        conn.close()

    return len(rows)