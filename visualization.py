"""Модуль визуализации.

Строит графики по результатам анализа и прогнозирования.
Функции возвращают Figure; save_figure сохраняет графики в PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import MONTHS_RU, fmt_int


BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SOFT, GRID = "#1f2933", "#52606d", "#e4e7eb"

MODEL_STYLE = {
    "Сезонная наивная": (INK_SOFT, "--"),
    "Регрессия": (BLUE, "-"),
    "SARIMAX": (ORANGE, "-"),
    "Хольт — Винтерс": (GREEN, "-"),
}


def _style(ax, ylabel: str | None = None, title: str | None = None) -> None:
    """Единое оформление осей."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.tick_params(colors=INK_SOFT, labelsize=9)

    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)

    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)


def _thousands(ax) -> None:
    """Ось Y в тысячах человек."""
    ax.yaxis.set_major_formatter(
        lambda value, _: f"{value / 1000:,.0f}".replace(",", " ")
    )


def _dates(index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return index.to_timestamp()


def save_figure(fig, path: str | Path) -> Path:
    """Сохранить график в PNG и освободить память."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return path


def plot_series(
    series: pd.Series,
    fitted: pd.Series,
    anomalies: pd.DataFrame,
    title: str,
):
    """Фактический ряд, сезонная модель и отмеченные аномалии."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = _dates(series.index)

    ax.plot(
        x,
        series.to_numpy(),
        color=BLUE,
        linewidth=1.6,
        marker="o",
        markersize=3,
    )
    ax.plot(
        _dates(fitted.index),
        fitted.to_numpy(),
        color=INK_SOFT,
        linewidth=1.2,
        linestyle="--",
    )

    if not anomalies.empty:
        pts = anomalies.index
        ax.scatter(
            _dates(pts),
            series.loc[pts].to_numpy(),
            s=60,
            facecolor="none",
            edgecolor=ORANGE,
            linewidth=1.6,
            zorder=5,
        )
        ax.annotate(
            "аномалия",
            (_dates(pts)[0], series.loc[pts[0]]),
            xytext=(8, 10),
            textcoords="offset points",
            color=INK_SOFT,
            fontsize=9,
        )

    ax.text(
        x[-1], series.iloc[-1], "  факт",
        color=INK_SOFT, fontsize=9, va="center",
    )
    ax.text(
        x[-1], fitted.iloc[-1], "  модель",
        color=INK_SOFT, fontsize=9, va="center",
    )

    _thousands(ax)
    _style(ax, "тыс. человек", title)
    ax.margins(x=0.08)
    return fig


def plot_seasonality(seasonal_index: pd.Series, title: str):
    """Сезонные индексы относительно геометрического среднего уровней месяцев."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = [month[:3] for month in MONTHS_RU]
    colours = [
        BLUE if value >= 1 else "#9bbbe6"
        for value in seasonal_index
    ]

    ax.bar(labels, seasonal_index.to_numpy(), color=colours, width=0.65)
    ax.axhline(1.0, color=INK_SOFT, linewidth=0.8)

    for i, value in enumerate(seasonal_index):
        ax.text(
            i, value + 0.02, f"{value:.2f}",
            ha="center", fontsize=8, color=INK_SOFT,
        )

    _style(ax, "сезонный индекс", title)
    return fig


def plot_residuals(
    residuals: pd.Series,
    anomalies: pd.DataFrame,
    title: str,
):
    """Остатки сезонной модели в процентах от модели."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    pct = (np.exp(residuals) - 1.0) * 100.0
    colours = [
        ORANGE if period in anomalies.index else BLUE
        for period in residuals.index
    ]

    ax.bar(
        _dates(residuals.index),
        pct.to_numpy(),
        width=22,
        color=colours,
    )
    ax.axhline(0, color=INK_SOFT, linewidth=0.8)
    _style(ax, "отклонение от модели, %", title)
    return fig


def plot_lag_correlations(
    lags: pd.DataFrame,
    factor_label: str,
    title: str,
):
    """Корреляция остатков с фактором при разных сдвигах по времени."""
    fig, ax = plt.subplots(figsize=(9, 3.6))

    ax.bar(lags.index, lags["corr"].to_numpy(), color=BLUE, width=0.6)
    ax.axhline(0, color=INK_SOFT, linewidth=0.8)
    ax.set_xticks(list(lags.index))
    ax.set_xlabel(
        f"сдвиг фактора «{factor_label}», месяцев назад",
        color=INK_SOFT,
        fontsize=9,
    )

    _style(ax, "коэффициент корреляции", title)
    return fig


def plot_metrics(metrics: pd.DataFrame, title: str):
    """Сравнение моделей по средней абсолютной процентной ошибке."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(metrics.index)
    colours = [
        MODEL_STYLE.get(name, (BLUE, "-"))[0]
        for name in names
    ]

    ax.barh(
        names[::-1],
        metrics["MAPE"].to_numpy()[::-1],
        color=colours[::-1],
        height=0.55,
    )

    for i, value in enumerate(metrics["MAPE"].to_numpy()[::-1]):
        ax.text(
            value + 0.15, i, f"{value:.1f} %",
            va="center", fontsize=9, color=INK,
        )

    _style(ax, None, title)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("MAPE, %", color=INK_SOFT, fontsize=9)
    ax.set_xlim(0, float(metrics["MAPE"].max()) * 1.2)
    return fig


def plot_error_by_horizon(by_horizon: pd.DataFrame, title: str):
    """Ошибка моделей в зависимости от шага прогноза."""
    fig, ax = plt.subplots(figsize=(8, 3.8))

    for name in by_horizon.columns:
        colour, dash = MODEL_STYLE.get(name, (BLUE, "-"))
        ax.plot(
            by_horizon.index,
            by_horizon[name].to_numpy(),
            color=colour,
            linestyle=dash,
            linewidth=1.6,
            marker="o",
            markersize=3.5,
        )
        ax.text(
            by_horizon.index[-1] + 0.08,
            by_horizon[name].iloc[-1],
            name,
            color=INK_SOFT,
            fontsize=9,
            va="center",
        )

    ax.set_xticks(list(by_horizon.index))
    ax.set_xlabel(
        "шаг прогноза, месяцев вперёд",
        color=INK_SOFT,
        fontsize=9,
    )
    ax.set_xlim(
        by_horizon.index.min() - 0.2,
        by_horizon.index.max() + 2.6,
    )

    _style(ax, "MAPE, %", title)
    return fig


def plot_backtest(
    backtest: pd.DataFrame,
    series: pd.Series,
    model: str,
    title: str,
):
    """Исторические прогнозы на один шаг и фактические значения."""
    fig, ax = plt.subplots(figsize=(9, 3.8))

    ax.plot(
        _dates(series.index),
        series.to_numpy(),
        color=BLUE,
        linewidth=1.6,
    )

    part = backtest[
        (backtest["model"] == model) & (backtest["h"] == 1)
    ].sort_values("period")

    ax.plot(
        _dates(pd.PeriodIndex(part["period"])),
        part["forecast"].to_numpy(),
        color=ORANGE,
        linewidth=1.4,
        marker="o",
        markersize=3,
    )
    ax.text(
        _dates(series.index)[-1],
        series.iloc[-1],
        "  факт",
        color=INK_SOFT,
        fontsize=9,
        va="center",
    )
    ax.text(
        _dates(pd.PeriodIndex(part["period"]))[-1],
        part["forecast"].iloc[-1],
        f"  {model}",
        color=INK_SOFT,
        fontsize=9,
        va="bottom",
    )

    _thousands(ax)
    _style(ax, "тыс. человек", title)
    ax.set_xlim(
        _dates(series.index)[0] - pd.Timedelta(days=20),
        _dates(series.index)[-1] + pd.Timedelta(days=150),
    )
    return fig


def plot_forecast(
    series: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    best: str,
    level: float,
    title: str,
    history_months: int = 30,
):
    """История и прогноз выбранной модели с интервалом неопределённости."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    hist = series.iloc[-history_months:]

    ax.plot(
        _dates(hist.index),
        hist.to_numpy(),
        color=BLUE,
        linewidth=1.8,
    )

    fc = forecasts[best]
    xs = _dates(fc.index.insert(0, hist.index[-1]))
    ys = np.concatenate([[hist.iloc[-1]], fc["forecast"].to_numpy()])

    ax.fill_between(
        _dates(fc.index),
        fc["lower"].to_numpy(),
        fc["upper"].to_numpy(),
        color=ORANGE,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        xs,
        ys,
        color=ORANGE,
        linewidth=1.8,
        marker="o",
        markersize=3.5,
    )
    ax.text(
        xs[-1],
        ys[-1],
        f"  {best}\n  (номинальный уровень {level * 100:.0f} %)",
        color=INK,
        fontsize=9,
        va="center",
    )

    ax.axvline(
        _dates(hist.index)[-1],
        color=GRID,
        linewidth=1,
    )

    _thousands(ax)
    _style(ax, "тыс. человек", title)
    ax.set_xlim(
        _dates(hist.index)[0] - pd.Timedelta(days=20),
        _dates(fc.index)[-1] + pd.Timedelta(days=120),
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    return fig


def plot_factor_effects(
    effects: pd.DataFrame,
    labels: dict[str, str],
    title: str,
):
    """Статистическая связь факторов с модельной оценкой.

    Масштаб изменения фактора задаётся в forecasting.regression_effects:
    для годового изменения курса — 0.01, для остальных факторов — 1.
    """
    fig, ax = plt.subplots(figsize=(8, 3.4))
    names = [labels.get(name, name) for name in effects.index]

    ax.barh(
        names[::-1],
        effects["effect_pct"].to_numpy()[::-1],
        color=BLUE,
        height=0.5,
    )
    ax.axvline(0, color=INK_SOFT, linewidth=0.8)

    _style(ax, None, title)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel(
        "изменение модельной оценки, %; USD: +1 п. п., прочие: +1 ед.",
        color=INK_SOFT,
        fontsize=9,
    )
    return fig


def build_all(
    out_dir: str | Path,
    series: pd.Series,
    season: dict,
    anomalies: pd.DataFrame,
    lags: pd.DataFrame,
    forecast_result,
    level: float,
    target_label: str,
) -> dict[str, Path]:
    """Построить и сохранить графики. Возвращает словарь путей."""
    out = Path(out_dir)
    paths: dict[str, Path] = {}

    def put(name, fig):
        paths[name] = save_figure(fig, out / f"{name}.png")

    put(
        "01_series",
        plot_series(
            series,
            season["fitted"],
            anomalies,
            f"{target_label}: факт и сезонная модель",
        ),
    )
    put(
        "02_seasonality",
        plot_seasonality(
            season["seasonal_index"],
            "Сезонный индекс по месяцам",
        ),
    )
    put(
        "03_residuals",
        plot_residuals(
            season["residuals"],
            anomalies,
            "Что сезонность и тренд не объяснили",
        ),
    )

    if not lags.empty and lags["corr"].notna().any():
        put(
            "04_lag_fx",
            plot_lag_correlations(
                lags,
                "курс доллара, годовое изменение",
                "Связь остатков с курсом при разных сдвигах",
            ),
        )

    put(
        "05_models_mape",
        plot_metrics(
            forecast_result.metrics,
            "Точность моделей на скользящей проверке",
        ),
    )
    put(
        "06_error_horizon",
        plot_error_by_horizon(
            forecast_result.by_horizon,
            "Ошибка по шагам прогноза",
        ),
    )
    put(
        "07_backtest",
        plot_backtest(
            forecast_result.backtest,
            series,
            forecast_result.best,
            f"Исторические прогнозы на 1 шаг: {forecast_result.best}",
        ),
    )
    put(
        "08_forecast",
        plot_forecast(
            series,
            forecast_result.forecasts,
            forecast_result.best,
            level,
            f"{target_label}: прогноз",
        ),
    )

    if not forecast_result.effects.empty:
        labels = {
            "nonworking_dev": "нерабочие дни (отклонение от нормы)",
            "fx_usd_yoy_lag7": "годовое изменение USD: +1 п. п. (лаг 7 мес.)",
            "sochi_rainy_days_anom": "дождливые дни, Сочи (откл. от нормы)",
            "anapa_sunshine_h_anom": "солнечные часы, Анапа (откл. от нормы)",
        }
        put(
            "09_factor_effects",
            plot_factor_effects(
                forecast_result.effects,
                labels,
                "Коэффициенты регрессии: статистические связи",
            ),
        )

    return paths


__all__ = ["build_all", "save_figure", "fmt_int"]