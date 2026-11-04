"""Погода: Open-Meteo Archive API (реанализ ERA5).

Открытый источник, ключ не требуется, лицензия CC BY 4.0,
история с 1940 г., суточная детализация.

Важно для прогноза: фактическая погода на горизонте прогнозирования
НЕ известна. Поэтому в модель эти данные входят либо как отклонение
от климатической нормы прошлых лет (для ретроспективного анализа),
либо через сценарии. Прямая подстановка факта = утечка из будущего.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from . import http

log = logging.getLogger(__name__)

# ERA5 публикуется с задержкой около пяти суток
PUBLICATION_LAG_DAYS = 6


def fetch_point(base_url: str, latitude: float, longitude: float,
                start: str, end: str, daily: list[str], timezone: str,
                **net) -> list[tuple]:
    """Вернуть строки (date, temp_mean, temp_max, precip_sum, sunshine_s)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(daily),
        "timezone": timezone,
    }
    response = http.get(base_url, params, **net)
    payload = response.json()

    if "daily" not in payload:
        raise http.SourceUnavailable(
            f"Open-Meteo вернул ответ без блока daily: {payload}"
        )

    block = payload["daily"]
    dates = block.get("time", [])
    columns = [block.get(name, [None] * len(dates)) for name in daily]

    rows = []
    for index, day in enumerate(dates):
        rows.append((day, *(column[index] for column in columns)))
    return rows


def effective_end(requested_end: str) -> str:
    """Не запрашивать даты, для которых реанализ ещё не опубликован."""
    limit = date.today() - timedelta(days=PUBLICATION_LAG_DAYS)
    return min(requested_end, limit.isoformat())
