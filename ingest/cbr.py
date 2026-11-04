"""Курсы валют: Банк России, сервис XML_dynamic.

Открытый источник, ключ не требуется. Отдаёт XML.
Особенности, которые надо учитывать:
  * даты в параметрах запроса — в формате ДД/ММ/ГГГГ;
  * дробная часть значения отделяется ЗАПЯТОЙ;
  * курс даётся за номинал (например, за 10 турецких лир),
    поэтому приводим к одной единице валюты;
  * курс публикуется только по рабочим дням.
"""

from __future__ import annotations

import logging
from datetime import date
from xml.etree import ElementTree

from . import http

log = logging.getLogger(__name__)


def _to_float(text: str) -> float:
    return float(text.replace(",", "."))


def _ru_date(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{day}/{month}/{year}"


def fetch_currency(base_url: str, code: str, start: str, end: str,
                   **net) -> list[tuple]:
    """Вернуть строки (date_iso, rate_per_unit)."""
    params = {
        "date_req1": _ru_date(start),
        "date_req2": _ru_date(end),
        "VAL_NM_RQ": code,
    }
    response = http.get(base_url, params, **net)

    try:
        root = ElementTree.fromstring(response.content)
    except ElementTree.ParseError as exc:
        raise http.SourceUnavailable(f"ЦБ РФ: не удалось разобрать XML: {exc}")

    rows = []
    for record in root.findall("Record"):
        raw_date = record.get("Date")           # ДД.ММ.ГГГГ
        value_node = record.find("Value")
        nominal_node = record.find("Nominal")
        if raw_date is None or value_node is None or value_node.text is None:
            continue

        day, month, year = raw_date.split(".")
        iso = f"{year}-{month}-{day}"

        value = _to_float(value_node.text)
        nominal = 1.0
        if nominal_node is not None and nominal_node.text:
            nominal = _to_float(nominal_node.text)

        rows.append((iso, value / nominal))

    if not rows:
        log.warning("ЦБ РФ: за период %s..%s не вернулось ни одной записи "
                    "(валюта %s)", start, end, code)
    return rows
