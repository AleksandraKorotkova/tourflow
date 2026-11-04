"""Разбор месячного файла Росстата по средствам размещения.

Особенность источника: значения публикуются НАРАСТАЮЩИМ ИТОГОМ внутри года
(январь; январь-февраль; январь-март; ...). Здесь они приводятся к месячным
величинам разностями соседних колонок. Январь берётся как есть.

Ещё одна особенность: с 2026 г. подпись региона содержит номер сноски
("Краснодарский край 2" — данные публикуются с учётом ФТ "Сириус"),
поэтому сопоставление идёт по префиксу, а факт наличия сноски фиксируется.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from openpyxl import load_workbook

log = logging.getLogger(__name__)

HEADER_ROW = 4          # строка с подписями периодов (1-индексация openpyxl)
SHEET_RE = re.compile(r"^(?P<prefix>[\d.]+?)\.(?P<year>\d{4})$")


def file_hash(path: str | Path) -> str:
    """Хеш файла — по нему определяется, обновилась ли выгрузка."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decumulate(values: list[float | None]) -> list[float | None]:
    """Из ряда нарастающим итогом получить месячные значения."""
    result: list[float | None] = []
    previous: float | None = None
    for current in values:
        if current is None:
            result.append(None)
            continue
        result.append(current if previous is None else current - previous)
        previous = current
    return result


def parse(path: str | Path, region_prefix: str,
          indicators: dict[str, str]) -> list[dict]:
    """Вернуть список записей {indicator, period, value, footnote}."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    records: list[dict] = []

    for sheet_name in workbook.sheetnames:
        match = SHEET_RE.match(sheet_name)
        if not match:
            continue
        prefix = match.group("prefix")
        year = int(match.group("year"))
        if prefix not in indicators:
            continue

        rows = list(workbook[sheet_name].iter_rows(values_only=True))
        if len(rows) < HEADER_ROW:
            continue

        n_periods = sum(1 for cell in rows[HEADER_ROW - 1][1:] if cell)

        for row in rows:
            label = str(row[0]).strip() if row[0] else ""
            if not label.startswith(region_prefix):
                continue
            # отсечь строку-сноску вида "2 Данные по Краснодарскому краю..."
            if len(label) > len(region_prefix) + 4:
                continue

            has_footnote = label != region_prefix
            raw = list(row[1:1 + n_periods])
            monthly = _decumulate(raw)

            for index, value in enumerate(monthly, start=1):
                if value is None:
                    continue
                if value < 0:
                    log.warning(
                        "%s %s-%02d: отрицательное значение после разности (%s). "
                        "Возможен ретроспективный пересчёт.",
                        indicators[prefix], year, index, value,
                    )
                records.append({
                    "indicator": indicators[prefix],
                    "period": f"{year}-{index:02d}",
                    "value": float(value),
                    "footnote": has_footnote,
                })
            break

    workbook.close()
    records.sort(key=lambda item: (item["indicator"], item["period"]))
    return records
