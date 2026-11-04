"""Модуль анализа данных.

Разведочный анализ ряда и оценка связи с внешними факторами:
  * сезонная составляющая и тренд;
  * аномальные наблюдения;
  * корреляция внешних факторов с тем, что сезонность не объясняет;
  * годовая сводка и текстовые выводы.

Модуль работает с уже подготовленной витриной (processing.prepare) и
ничего не читает с диска и не рисует: на вход — таблицы и ряды, на выход —
таблицы, словари и строки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
             "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def fmt_int(value: float) -> str:
    """Число с пробелами между разрядами: 1587267 -> '1 587 267'."""
    return f"{value:,.0f}".replace(",", " ")


def month_name(month: int) -> str:
    """Название месяца по номеру: 7 -> 'июль'."""
    return MONTHS_RU[month - 1]


def fmt_period(period: pd.Period) -> str:
    """Период словами: 2022-07 -> 'июль 2022'."""
    return f"{MONTHS_RU[period.month - 1]} {period.year}"


# ---------------------------------------------------------------------------
# Сезонность и тренд
# ---------------------------------------------------------------------------

def fit_seasonal_trend(series: pd.Series) -> dict:
    """Оценить сезонную составляющую и линейный тренд.

    Модель на логарифме показателя: ln y = сезонный уровень месяца + b * t.
    Логарифм нужен, потому что размах сезонных колебаний растёт вместе с
    уровнем ряда (мультипликативная сезонность).

    Возвращает словарь:
      fitted         — подогнанные значения (в исходных единицах);
      residuals      — остатки модели в логарифмической шкале (то, что
                       сезонность и тренд не объяснили);
      r2             — доля объяснённой изменчивости логарифма;
      seasonal_index — сезонный индекс по месяцам (1.0 = средний месяц);
      trend_pct_year — годовой темп тренда, %.
    """
    log_y = np.log(series.to_numpy(dtype=float))
    t = np.arange(len(series), dtype=float)
    months = series.index.month.to_numpy()

    design = np.column_stack(
        [(months == m).astype(float) for m in range(1, 13)] + [t]
    )
    coef, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    fitted_log = design @ coef
    residuals = log_y - fitted_log

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))

    month_levels = coef[:12]
    seasonal_index = pd.Series(np.exp(month_levels - month_levels.mean()),
                               index=range(1, 13), name="seasonal_index")

    return {
        "fitted": pd.Series(np.exp(fitted_log), index=series.index),
        "residuals": pd.Series(residuals, index=series.index, name="residual"),
        "r2": 1.0 - ss_res / ss_tot,
        "seasonal_index": seasonal_index,
        "trend_pct_year": (np.exp(coef[12] * 12) - 1.0) * 100.0,
    }


# ---------------------------------------------------------------------------
# Аномалии
# ---------------------------------------------------------------------------

def detect_anomalies(residuals: pd.Series, threshold: float = 2.5) -> pd.DataFrame:
    """Найти наблюдения, нетипично далёкие от сезонной модели.

    Используется робастная оценка масштаба (медианное абсолютное отклонение,
    MAD): среднее и стандартное отклонение сами искажаются выбросами.
    Остаток считается аномальным, если его робастный z-score по модулю
    больше threshold.

    Столбцы результата: residual (в логарифмах), deviation_pct (отклонение
    факта от модели, %), z.
    """
    centre = residuals.median()
    mad = (residuals - centre).abs().median()
    scale = 1.4826 * mad if mad > 0 else residuals.std()
    z = (residuals - centre) / scale

    flagged = z[z.abs() > threshold]
    result = pd.DataFrame({
        "residual": residuals[flagged.index],
        "deviation_pct": (np.exp(residuals[flagged.index]) - 1.0) * 100.0,
        "z": flagged,
    })
    return result.reindex(result["z"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# Связь с внешними факторами
# ---------------------------------------------------------------------------

def factor_correlations(residuals: pd.Series, frame: pd.DataFrame,
                        factors: list[str]) -> pd.DataFrame:
    """Корреляция остатков модели с внешними факторами в том же месяце.

    Остатки — это изменчивость показателя, которую сезонность и тренд не
    объяснили; именно с ней имеет смысл сопоставлять внешние факторы.
    Столбцы результата: corr, n (число пар наблюдений).
    """
    rows = []
    for name in factors:
        if name not in frame.columns:
            continue
        joined = pd.concat([residuals, frame[name].reindex(residuals.index)],
                           axis=1).dropna()
        if len(joined) < 10:
            continue
        rows.append({"factor": name,
                     "corr": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
                     "n": int(len(joined))})
    table = pd.DataFrame(rows).set_index("factor")
    return table.reindex(table["corr"].abs().sort_values(ascending=False).index)


def lag_correlations(residuals: pd.Series, factor: pd.Series,
                     max_lag: int = 12) -> pd.DataFrame:
    """Корреляция остатков с фактором, сдвинутым на 0..max_lag месяцев назад.

    Решение о поездке принимается заранее, поэтому влияние, например,
    валютного курса может проявляться не сразу. Лаг k означает: значение
    фактора k месяцев назад сопоставляется с остатком текущего месяца.
    """
    rows = []
    for lag in range(max_lag + 1):
        shifted = factor.shift(lag).reindex(residuals.index)
        joined = pd.concat([residuals, shifted], axis=1).dropna()
        corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1])) if len(joined) >= 10 else np.nan
        rows.append({"lag": lag, "corr": corr, "n": int(len(joined))})
    return pd.DataFrame(rows).set_index("lag")


# ---------------------------------------------------------------------------
# Сводки
# ---------------------------------------------------------------------------

def yearly_summary(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Годовая сводка: сумма показателя по полным годам и доля санаториев."""
    series = frame[target].dropna()
    years = series.groupby(series.index.year)
    summary = pd.DataFrame({"total": years.sum(), "months": years.count()})
    summary = summary[summary["months"] == 12].drop(columns="months")
    if "sanatorium_share" in frame.columns:
        share = frame["sanatorium_share"].dropna()
        summary["sanatorium_share_pct"] = (
            share.groupby(share.index.year).mean().reindex(summary.index) * 100.0
        )
    return summary


# ---------------------------------------------------------------------------
# Текстовые выводы
# ---------------------------------------------------------------------------

def make_conclusions(series: pd.Series, model: dict, anomalies: pd.DataFrame,
                     correlations: pd.DataFrame, metrics: pd.DataFrame,
                     forecast: pd.DataFrame, best_model: str,
                     horizon_label: str) -> list[str]:
    """Сформировать текстовые аналитические выводы.

    Каждый вывод сопровождается указанием на его границы: связь факторов —
    статистическая, а не причинная; качество оценено на ограниченной истории.
    """
    lines: list[str] = []

    low, high = series.idxmin(), series.idxmax()
    lines.append(
        f"Показатель за {fmt_period(series.index[0])} — {fmt_period(series.index[-1])} "
        f"({len(series)} мес.): от {fmt_int(series.min())} ({fmt_period(low)}) до "
        f"{fmt_int(series.max())} ({fmt_period(high)}) человек, размах в "
        f"{series.max() / series.min():.1f} раза."
    )

    index = model["seasonal_index"]
    peak = int(index.idxmax())
    trough = int(index.idxmin())
    lines.append(
        f"Сезонная составляющая и тренд объясняют {model['r2'] * 100:.0f} % изменчивости "
        f"логарифма показателя. Пик приходится на {month_name(peak)} "
        f"(индекс {index.max():.2f}), минимум — на {month_name(trough)} "
        f"(индекс {index.min():.2f}); тренд за пределами сезонности — "
        f"{model['trend_pct_year']:+.1f} % в год."
    )

    if anomalies.empty:
        lines.append("Аномальных наблюдений не выявлено.")
    else:
        top = anomalies.iloc[0]
        lines.append(
            f"Выявлено аномальных наблюдений: {len(anomalies)}. Наиболее сильное — "
            f"{fmt_period(anomalies.index[0])}: факт на {top['deviation_pct']:+.0f} % "
            f"относительно сезонной модели. Наблюдения сохранены без изменения. "
            f"Причины отклонений требуют дополнительного анализа; обнаружение "
            f"аномалии само по себе не устанавливает её причину."
        )

    if not correlations.empty:
        best = correlations.iloc[0]
        factor_labels = {
            "nonworking_dev": "число нерабочих дней: отклонение от нормы",
            "fx_usd_yoy": "годовое изменение курса доллара",
            "sochi_rainy_days_anom": "дождливые дни в Сочи: отклонение от нормы",
            "sochi_sunshine_h_anom": "солнечные часы в Сочи: отклонение от нормы",
            "anapa_sunshine_h_anom": "солнечные часы в Анапе: отклонение от нормы",
            "anapa_temp_mean_anom": "температура в Анапе: отклонение от нормы",
        }
        factor_name = factor_labels.get(
            correlations.index[0], correlations.index[0]
        )
        lines.append(
            f"Среди внешних факторов наиболее тесно с остатками модели связан "
            f"«{factor_name}» (коэффициент корреляции {best['corr']:+.2f}). "
            f"Связь статистическая и не доказывает причинно-следственной зависимости."
        )

    if not metrics.empty:
        top_model = metrics["MAPE"].idxmin()
        text = (f"По результатам скользящей проверки наименьшую среднюю процентную "
                f"ошибку даёт модель «{top_model}» ({metrics.loc[top_model, 'MAPE']:.1f} %).")
        base = "Сезонная наивная"
        if base in metrics.index and top_model != base:
            gain = metrics.loc[base, "MAPE"] - metrics.loc[top_model, "MAPE"]
            text += (f" Базовая сезонная наивная модель ошибается на "
                     f"{metrics.loc[base, 'MAPE']:.1f} %, то есть выигрыш составляет "
                     f"{gain:.1f} п. п.")
        elif top_model == base:
            text += (" Более сложные модели не превзошли базовую: на имеющейся "
                     "истории (несколько сезонов) сезонное повторение прошлого года "
                     "оказывается трудно обыграть.")
        lines.append(text)

    if not forecast.empty:
        first, last = forecast.index[0], forecast.index[-1]
        peak_row = forecast["forecast"].idxmax()
        lines.append(
            f"Прогноз модели «{best_model}» на {horizon_label} "
            f"({fmt_period(first)} — {fmt_period(last)}): максимум в "
            f"{fmt_period(peak_row)} — около {fmt_int(forecast.loc[peak_row, 'forecast'])} "
            f"человек (границы интервала: {fmt_int(forecast.loc[peak_row, 'lower'])} — "
            f"{fmt_int(forecast.loc[peak_row, 'upper'])})."
        )
    return lines

