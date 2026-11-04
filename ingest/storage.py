"""Слой хранения: PostgreSQL + журнал загрузок.

Здесь реализованы два требования методички:
  * идемпотентность — запись через INSERT ... ON CONFLICT DO UPDATE,
    поэтому повторный запуск не создаёт дублей и не меняет результат;
  * журнал загрузок — таблица load_log, куда пишется каждый запуск.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Sequence

import psycopg2
import psycopg2.extras

SCHEMA = """
-- Целевые показатели Росстата: длинный формат
CREATE TABLE IF NOT EXISTS rosstat_monthly (
    region      TEXT NOT NULL,
    indicator   TEXT NOT NULL,
    period      TEXT NOT NULL,          -- 'YYYY-MM'
    value       DOUBLE PRECISION,
    is_cumulative_source INTEGER NOT NULL DEFAULT 1,
    loaded_at   TEXT NOT NULL,
    PRIMARY KEY (region, indicator, period)
);

-- Суточная погода по опорным точкам
CREATE TABLE IF NOT EXISTS weather_daily (
    point       TEXT NOT NULL,
    date        TEXT NOT NULL,          -- 'YYYY-MM-DD'
    temp_mean   DOUBLE PRECISION,
    temp_max    DOUBLE PRECISION,
    precip_sum  DOUBLE PRECISION,
    sunshine_s  DOUBLE PRECISION,
    loaded_at   TEXT NOT NULL,
    PRIMARY KEY (point, date)
);

-- Курсы валют ЦБ РФ
CREATE TABLE IF NOT EXISTS fx_daily (
    currency    TEXT NOT NULL,
    date        TEXT NOT NULL,
    rate        DOUBLE PRECISION NOT NULL,
    loaded_at   TEXT NOT NULL,
    PRIMARY KEY (currency, date)
);

-- Производственный календарь
CREATE TABLE IF NOT EXISTS calendar_daily (
    date        TEXT NOT NULL PRIMARY KEY,
    day_type    INTEGER NOT NULL,       -- 0 рабочий, 1 нерабочий, 2 сокращённый, 4 covid
    loaded_at   TEXT NOT NULL
);

-- Журнал загрузок
CREATE TABLE IF NOT EXISTS load_log (
    run_id      TEXT NOT NULL,
    source      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,          -- success | failed | skipped | running
    rows_read   INTEGER DEFAULT 0,
    rows_written INTEGER DEFAULT 0,
    period_from TEXT,
    period_to   TEXT,
    message     TEXT,
    PRIMARY KEY (run_id, source)
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, pg_config: dict):
        """pg_config — словарь с ключами host, port, dbname, user, password.

        База данных (пустая) должна существовать заранее — проще всего
        создать её через DBeaver: правой кнопкой по соединению →
        Create New Database. Таблицы внутри неё создаёт сам этот класс
        при первом подключении.
        """
        self.conn = psycopg2.connect(**pg_config)
        self.conn.autocommit = False
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------- чтение произвольных запросов ----------

    def query(self, sql: str, params: Sequence = ()) -> list[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description is None:
                return []
            return cur.fetchall()

    # ---------- запись данных (идемпотентная) ----------

    def upsert(self, table: str, columns: Sequence[str],
               rows: Iterable[Sequence], key_columns: Sequence[str]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        placeholders = ", ".join("%s" for _ in columns)
        updates = ", ".join(
            f"{c}=EXCLUDED.{c}" for c in columns if c not in key_columns
        )
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(key_columns)}) DO UPDATE SET {updates}"
        )
        with self.tx() as conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)
        return len(rows)

    # ---------- инкрементальность ----------

    def max_value(self, table: str, column: str,
                  where: str = "", params: Sequence = ()) -> str | None:
        """Последнее загруженное значение — точка возобновления загрузки."""
        sql = f"SELECT MAX({column}) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    # ---------- журнал ----------

    def log_start(self, run_id: str, source: str) -> None:
        with self.tx() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO load_log (run_id, source, started_at, status) "
                "VALUES (%s, %s, %s, 'running') "
                "ON CONFLICT (run_id, source) DO UPDATE "
                "SET started_at = EXCLUDED.started_at, status = 'running'",
                (run_id, source, utcnow()),
            )

    def log_finish(self, run_id: str, source: str, status: str,
                   rows_read: int = 0, rows_written: int = 0,
                   period_from: str | None = None, period_to: str | None = None,
                   message: str | None = None) -> None:
        with self.tx() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE load_log SET finished_at=%s, status=%s, rows_read=%s, "
                "rows_written=%s, period_from=%s, period_to=%s, message=%s "
                "WHERE run_id=%s AND source=%s",
                (utcnow(), status, rows_read, rows_written,
                 period_from, period_to, message, run_id, source),
            )
