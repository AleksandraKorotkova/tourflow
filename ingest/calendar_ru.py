"""Производственный календарь: isdayoff.ru.

Открытый источник, ключ не требуется. Отдаёт строку из символов —
по одному на каждый день запрошенного периода:
    0 — рабочий день
    1 — нерабочий (выходной или праздник)
    2 — сокращённый рабочий день (при pre=1)
    4 — нерабочий по особому режиму (covid=1)

Этот фактор известен заранее на весь горизонт прогноза, что снимает
требование методички о доступности значения на момент построения прогноза.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from . import http

log = logging.getLogger(__name__)


def fetch_year(base_url: str, year: int, country: str = "ru",
               **net) -> list[tuple]:
    """Вернуть строки (date_iso, day_type) за календарный год."""
    params = {"year": year, "cc": country, "pre": 1}
    response = http.get(base_url, params, **net)
    payload = response.text.strip()

    # Сервис при ошибке отвечает кодом в теле: 100/101/199
    if payload in {"100", "101", "199"} or not payload.isdigit():
        raise http.SourceUnavailable(
            f"isdayoff вернул код ошибки для {year}: {payload!r}"
        )

    start = date(year, 1, 1)
    expected = (date(year + 1, 1, 1) - start).days
    if len(payload) != expected:
        raise http.SourceUnavailable(
            f"isdayoff: за {year} ожидалось {expected} символов, "
            f"получено {len(payload)}"
        )

    rows = []
    for offset, symbol in enumerate(payload):
        current = start + timedelta(days=offset)
        rows.append((current.isoformat(), int(symbol)))
    return rows
