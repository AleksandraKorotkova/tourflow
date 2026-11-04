"""Модуль прогнозирования.

Четыре модели прогноза месячного показателя, единая схема их проверки и
оценка неопределённости:

  * «Сезонная наивная» — базовая: значение того же месяца год назад;
  * «Регрессия» — линейная модель на сезонных индикаторах, тренде и внешних
    факторах (коэффициенты интерпретируемы);
  * «SARIMAX» — сезонная авторегрессионная модель с внешними регрессорами;
  * «Хольт — Винтерс» — экспоненциальное сглаживание с сезонностью.

Проверка — скользящая (rolling origin): модель обучается на первых N месяцах,
прогнозирует следующие H, окно сдвигается на месяц вперёд, и так до конца
ряда. Будущее при обучении никогда не используется. Метрики считаются по
всем прогнозам, сделанным «вслепую».

Внешние факторы подаются в модели так, как они были бы известны на момент
прогноза: календарь — всегда; курс — со сдвигом (значение месяца t−lag);
погода — фактическая для прошлого и нулевое отклонение (норма) для будущего.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:  # SARIMAX и Хольт — Винтерс требуют statsmodels
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:  # без библиотеки работают две первые модели
    HAS_STATSMODELS = False

log = logging.getLogger(__name__)

BASELINE = "Сезонная наивная"
SEASON = 12


@dataclass
class ForecastSettings:
    """Параметры прогнозирования (читаются из config.yaml)."""

    horizon: int = 6                  # горизонт прогноза, месяцев
    initial_train: int = 30           # длина первого обучающего окна
    interval_level: float = 0.90      # уровень доверительного интервала
    fx_lag: int = 7                   # сдвиг курса, месяцев
    calendar_factors: tuple = ("nonworking_dev",)
    fx_factors: tuple = ("fx_usd_yoy",)
    weather_factors: tuple = ("sochi_rainy_days_anom", "anapa_sunshine_h_anom")

    @classmethod
    def from_dict(cls, data: dict | None) -> "ForecastSettings":
        data = dict(data or {})
        for key in ("calendar_factors", "fx_factors", "weather_factors"):
            if key in data:
                data[key] = tuple(data[key])
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class ModelUnavailable(RuntimeError):
    """Модель недоступна (не установлена нужная библиотека)."""


# ---------------------------------------------------------------------------
# Разбиение по времени и внешние факторы
# ---------------------------------------------------------------------------

def time_split(series: pd.Series, end: int, horizon: int
               ) -> tuple[pd.Series, pd.Series]:
    """Разбить ряд по времени: обучение — первые `end` значений, проверка —
    следующие `horizon`. Порядок наблюдений никогда не нарушается."""
    return series.iloc[:end], series.iloc[end:end + horizon]


def build_exog(frame: pd.DataFrame, periods: pd.PeriodIndex, origin: pd.Period,
               settings: ForecastSettings) -> pd.DataFrame:
    """Внешние факторы для `periods` так, как они известны на момент `origin`.

    origin — последний месяц, для которого известно значение показателя.
      * календарь известен заранее; пустое значение (недостоверный год)
        заменяется нулём — «типичный месяц»;
      * курс: берётся значение за месяц (t − fx_lag); если оно позже origin
        или отсутствует — подставляется ноль (нет изменения);
      * погода: для месяцев до origin включительно — фактическое отклонение,
        для будущих — ноль (климатическая норма).
    """
    exog = pd.DataFrame(index=periods)

    for name in settings.calendar_factors:
        if name in frame.columns:
            exog[name] = frame[name].reindex(periods).fillna(0.0).to_numpy()

    for name in settings.fx_factors:
        if name in frame.columns:
            source = periods - settings.fx_lag
            values = frame[name].reindex(source).to_numpy(dtype=float)
            values = np.where(np.asarray(source <= origin), values, np.nan)
            exog[f"{name}_lag{settings.fx_lag}"] = np.nan_to_num(values, nan=0.0)

    for name in settings.weather_factors:
        if name in frame.columns:
            values = frame[name].reindex(periods).to_numpy(dtype=float)
            values = np.where(np.asarray(periods <= origin), values, 0.0)
            exog[name] = np.nan_to_num(values, nan=0.0)

    return exog


# ---------------------------------------------------------------------------
# Модели. Единая сигнатура: (y, X_train, X_future, horizon) -> прогноз (массив)
# ---------------------------------------------------------------------------

def seasonal_naive(y: pd.Series, X_train: pd.DataFrame, X_future: pd.DataFrame,
                   horizon: int) -> np.ndarray:
    """Базовая модель: прогноз = значение того же месяца год назад."""
    last_year = y.to_numpy(dtype=float)[-SEASON:]
    return np.array([last_year[i % SEASON] for i in range(horizon)])


def _regression_design(index: pd.PeriodIndex, t_start: int,
                       exog: pd.DataFrame) -> pd.DataFrame:
    """Матрица признаков регрессии: сезонные индикаторы месяцев 2–12, тренд
    (номер месяца по порядку) и внешние факторы."""
    design = pd.DataFrame(index=index)
    for month in range(2, 13):
        design[f"month_{month}"] = (index.month == month).astype(float)
    design["trend"] = np.arange(t_start, t_start + len(index), dtype=float)
    for name in exog.columns:
        design[name] = exog[name].to_numpy()
    return design


def regression(y: pd.Series, X_train: pd.DataFrame, X_future: pd.DataFrame,
               horizon: int) -> np.ndarray:
    """Линейная регрессия на логарифме показателя.

    Признаки: сезонные индикаторы месяцев, линейный тренд, внешние факторы.
    Небольшая L2-регуляризация (alpha = 1e-3) стабилизирует решение, почти не
    меняя коэффициентов. Прогноз возвращается в исходных единицах.
    """
    train = _regression_design(y.index, 0, X_train)
    future_index = pd.period_range(y.index[-1] + 1, periods=horizon, freq="M")
    future = _regression_design(future_index, len(y), X_future)
    model = Ridge(alpha=1e-3).fit(train, np.log(y.to_numpy(dtype=float)))
    return np.exp(model.predict(future))


def regression_effects(y: pd.Series, X_train: pd.DataFrame) -> pd.DataFrame:
    """Вклад внешних факторов в модель регрессии.

    Для каждого фактора: коэффициент (в логарифмах) и эффект — на сколько
    процентов меняется показатель при росте фактора на единицу.
    """
    train = _regression_design(y.index, 0, X_train)
    model = Ridge(alpha=1e-3).fit(train, np.log(y.to_numpy(dtype=float)))
    rows = []
    for name, coef in zip(train.columns, model.coef_):
        if name in X_train.columns:
            step = 0.01 if name.startswith("fx_") else 1.0
            rows.append({
                "factor": name,
                "coef": float(coef),
                "factor_step": step,
                "effect_pct": float(np.expm1(coef * step) * 100.0),
            })
    return pd.DataFrame(rows).set_index("factor")


def _standardise(train: pd.DataFrame, future: pd.DataFrame
                 ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Привести внешние факторы к единому масштабу по обучающей выборке."""
    if train.shape[1] == 0:
        return None, None
    mean = train.mean()
    std = train.std(ddof=0).replace(0.0, 1.0)
    return ((train - mean) / std).to_numpy(), ((future - mean) / std).to_numpy()


def sarimax(y: pd.Series, X_train: pd.DataFrame, X_future: pd.DataFrame,
            horizon: int) -> np.ndarray:
    """SARIMAX(1,0,0)(0,1,0)₁₂ на логарифме показателя с внешними регрессорами.

    Сезонное дифференцирование убирает годовой цикл и тренд, авторегрессия
    первого порядка описывает остаточную инерцию, внешние факторы входят как
    экзогенные переменные.
    """
    if not HAS_STATSMODELS:
        raise ModelUnavailable("не установлена библиотека statsmodels")
    exog_train, exog_future = _standardise(X_train, X_future)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            np.log(y.to_numpy(dtype=float)), exog=exog_train,
            order=(1, 0, 0), seasonal_order=(0, 1, 0, SEASON),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
        forecast = fitted.forecast(steps=horizon, exog=exog_future)
    return np.exp(np.asarray(forecast, dtype=float))


def holt_winters(y: pd.Series, X_train: pd.DataFrame, X_future: pd.DataFrame,
                 horizon: int) -> np.ndarray:
    """Экспоненциальное сглаживание Хольта — Винтерса на логарифме показателя:
    уровень, затухающий тренд и аддитивная сезонность обновляются с каждым
    наблюдением; свежие данные весят больше старых."""
    if not HAS_STATSMODELS:
        raise ModelUnavailable("не установлена библиотека statsmodels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ExponentialSmoothing(
            np.log(y.to_numpy(dtype=float)), trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=SEASON,
            initialization_method="estimated",
        ).fit()
        forecast = fitted.forecast(horizon)
    return np.exp(np.asarray(forecast, dtype=float))


MODELS: dict[str, Callable] = {
    BASELINE: seasonal_naive,
    "Регрессия": regression,
    "SARIMAX": sarimax,
    "Хольт — Винтерс": holt_winters,
}


def available_models() -> dict[str, Callable]:
    """Модели, которые можно запустить в текущем окружении."""
    skip = set() if HAS_STATSMODELS else {"SARIMAX", "Хольт — Винтерс"}
    return {name: fn for name, fn in MODELS.items() if name not in skip}


# ---------------------------------------------------------------------------
# Скользящая проверка и метрики
# ---------------------------------------------------------------------------

def rolling_backtest(frame: pd.DataFrame, series: pd.Series,
                     settings: ForecastSettings,
                     models: dict[str, Callable] | None = None) -> pd.DataFrame:
    """Скользящая проверка моделей.

    Возвращает длинную таблицу: model, origin (последний месяц обучения),
    period, h (шаг прогноза, 1..horizon), actual, forecast.
    Если модель на каком-то окне упала с ошибкой, окно пропускается и это
    записывается в лог — остальные модели продолжают работу.
    """
    models = models or available_models()
    rows: list[dict] = []
    for end in range(settings.initial_train, len(series) - settings.horizon + 1):
        train, test = time_split(series, end, settings.horizon)
        origin = train.index[-1]
        X_train = build_exog(frame, train.index, origin, settings)
        X_test = build_exog(frame, test.index, origin, settings)
        for name, fn in models.items():
            try:
                predicted = fn(train, X_train, X_test, settings.horizon)
            except Exception as exc:  # noqa: BLE001 — устойчивость проверки
                log.warning("%s: окно до %s пропущено: %s", name, origin, exc)
                continue
            for step, (period, actual) in enumerate(test.items(), start=1):
                rows.append({"model": name, "origin": origin, "period": period,
                             "h": step, "actual": float(actual),
                             "forecast": float(predicted[step - 1])})
    return pd.DataFrame(rows)


def compute_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    """Метрики точности по моделям: MAPE (%) и RMSE (человек)."""
    err = backtest["actual"] - backtest["forecast"]
    work = backtest.assign(ape=err.abs() / backtest["actual"] * 100.0, sq=err ** 2)
    grouped = work.groupby("model")
    result = pd.DataFrame({"MAPE": grouped["ape"].mean(),
                           "RMSE": np.sqrt(grouped["sq"].mean()),
                           "n": grouped.size()})
    order = [m for m in MODELS if m in result.index]
    return result.loc[order]


def metrics_by_horizon(backtest: pd.DataFrame) -> pd.DataFrame:
    """MAPE (%) по шагам прогноза: строки — h, столбцы — модели."""
    work = backtest.assign(
        ape=(backtest["actual"] - backtest["forecast"]).abs() / backtest["actual"] * 100.0)
    table = work.pivot_table(index="h", columns="model", values="ape", aggfunc="mean")
    return table[[m for m in MODELS if m in table.columns]]


def best_model(metrics: pd.DataFrame) -> str:
    """Модель с наименьшей MAPE."""
    return str(metrics["MAPE"].idxmin())


# ---------------------------------------------------------------------------
# Итоговый прогноз и неопределённость
# ---------------------------------------------------------------------------

def error_scale(backtest: pd.DataFrame, model: str, horizon: int) -> pd.Series:
    """Масштаб ошибки модели по шагам прогноза: корень из среднего квадрата
    ошибки в логарифмической шкале. Если для шага мало наблюдений (< 8),
    используется общий масштаб по всем шагам."""
    part = backtest[backtest["model"] == model]
    log_err = np.log(part["actual"] / part["forecast"])
    pooled = float(np.sqrt((log_err ** 2).mean()))
    scale = {}
    for step in range(1, horizon + 1):
        sel = log_err[part["h"] == step]
        scale[step] = float(np.sqrt((sel ** 2).mean())) if len(sel) >= 8 else pooled
    return pd.Series(scale, name="sigma")


def final_forecast(frame: pd.DataFrame, series: pd.Series, model: str,
                   settings: ForecastSettings, backtest: pd.DataFrame
                   ) -> pd.DataFrame:
    """Прогноз на `horizon` месяцев после конца ряда с доверительным интервалом.

    Интервал строится по фактической ошибке модели при скользящей проверке:
    forecast · exp(± z · σ_h), где σ_h — масштаб ошибки для шага h, а z —
    квантиль нормального распределения для заданного уровня.
    """
    fn = MODELS[model]
    origin = series.index[-1]
    future = pd.period_range(origin + 1, periods=settings.horizon, freq="M")
    X_train = build_exog(frame, series.index, origin, settings)
    X_future = build_exog(frame, future, origin, settings)
    point = fn(series, X_train, X_future, settings.horizon)

    z = NormalDist().inv_cdf(0.5 + settings.interval_level / 2)
    sigma = error_scale(backtest, model, settings.horizon).to_numpy()
    return pd.DataFrame({
        "forecast": point,
        "lower": point * np.exp(-z * sigma),
        "upper": point * np.exp(z * sigma),
    }, index=future)


@dataclass
class ForecastResult:
    """Результат работы модуля прогнозирования."""

    backtest: pd.DataFrame
    metrics: pd.DataFrame
    by_horizon: pd.DataFrame
    best: str
    forecasts: dict[str, pd.DataFrame] = field(default_factory=dict)
    effects: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: list[str] = field(default_factory=list)


def run_forecasting(frame: pd.DataFrame, series: pd.Series,
                    settings: ForecastSettings) -> ForecastResult:
    """Полный цикл: проверка моделей, выбор лучшей, итоговые прогнозы."""
    models = available_models()
    skipped = [name for name in MODELS if name not in models]
    for name in skipped:
        log.warning("Модель «%s» пропущена: не установлена statsmodels "
                    "(pip install statsmodels)", name)

    backtest = rolling_backtest(frame, series, settings, models)
    if backtest.empty:
        raise RuntimeError("Скользящая проверка не дала результатов: "
                           "ряд слишком короткий для заданных параметров")
    metrics = compute_metrics(backtest)
    chosen = best_model(metrics)

    forecasts = {name: final_forecast(frame, series, name, settings, backtest)
                 for name in metrics.index}
    origin = series.index[-1]
    effects = regression_effects(series, build_exog(frame, series.index, origin, settings))
    return ForecastResult(backtest, metrics, metrics_by_horizon(backtest),
                          chosen, forecasts, effects, skipped)


def check_interval_coverage(
    backtest: pd.DataFrame,
    level: float = 0.90,
    min_errors: int = 8,
) -> pd.DataFrame:
    """Проверить интервалы по ошибкам, известным до начала прогноза."""
    rows = []
    z = NormalDist().inv_cdf(0.5 + level / 2)

    for model, part in backtest.groupby("model"):
        part = part.sort_values(["origin", "h"]).copy()

        for _, row in part.iterrows():
            # Используем только более ранние прогнозы, факты для которых
            # уже известны к текущему моменту прогнозирования.
            previous = part[
                (part["origin"] < row["origin"])
                & (part["period"] <= row["origin"])
                & (part["h"] == row["h"])
            ]

            if len(previous) < min_errors:
                continue

            errors = np.log(
                previous["actual"] / previous["forecast"]
            )
            sigma = float(np.sqrt((errors ** 2).mean()))
            lower = float(row["forecast"] * np.exp(-z * sigma))
            upper = float(row["forecast"] * np.exp(z * sigma))

            rows.append({
                "model": model,
                "origin": row["origin"],
                "period": row["period"],
                "h": int(row["h"]),
                "actual": float(row["actual"]),
                "forecast": float(row["forecast"]),
                "lower": lower,
                "upper": upper,
                "covered": bool(lower <= row["actual"] <= upper),
                "width": upper - lower,
                "calibration_n": len(previous),
            })

    return pd.DataFrame(rows, columns=[
        "model", "origin", "period", "h", "actual", "forecast",
        "lower", "upper", "covered", "width", "calibration_n",
    ])