"""Устойчивый HTTP-клиент.

Закрывает требование методички "устойчивость к сбоям": таймауты,
недоступность источника, ограничение частоты запросов (HTTP 429),
временные ошибки сервера (5xx) — всё это переживается повторами
с экспоненциальной задержкой.
"""

from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class SourceUnavailable(RuntimeError):
    """Источник недоступен после всех попыток."""


def get(url: str, params: dict | None = None, *,
        timeout: float = 30.0, max_retries: int = 4,
        backoff: float = 2.0, delay: float = 0.0) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            log.warning("Попытка %s/%s: сетевая ошибка: %s",
                        attempt, max_retries, exc)
        else:
            if response.status_code == 200:
                if delay:
                    time.sleep(delay)
                return response
            if response.status_code in RETRYABLE_STATUS:
                last_error = RuntimeError(f"HTTP {response.status_code}")
                log.warning("Попытка %s/%s: HTTP %s",
                            attempt, max_retries, response.status_code)
            else:
                # 4xx, кроме перечисленных, повторять бессмысленно
                raise SourceUnavailable(
                    f"{url}: HTTP {response.status_code}"
                )

        if attempt < max_retries:
            pause = backoff * (2 ** (attempt - 1))
            log.info("Повтор через %.1f c", pause)
            time.sleep(pause)

    raise SourceUnavailable(f"{url}: не удалось получить данные: {last_error}")
