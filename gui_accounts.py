"""Локальные учётные записи настольного приложения."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

ADMIN = "admin"
ANALYST = "analyst"
ROLES = {ADMIN: "Администратор", ANALYST: "Аналитик"}
ITERATIONS = 600_000


@dataclass(frozen=True)
class Account:
    login: str
    role: str

    def require_admin(self) -> None:
        if self.role != ADMIN:
            raise PermissionError("Операция доступна только администратору.")


class Accounts:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    login TEXT PRIMARY KEY,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'analyst')),
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    iterations INTEGER NOT NULL
                )
            """)

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            yield db

    @staticmethod
    def _hash(password: str, salt: bytes, iterations: int) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)

    def needs_setup(self) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM accounts LIMIT 1").fetchone() is None

    def authenticate(self, login: str, password: str) -> Account:
        login = login.strip().casefold()
        with self.connect() as db:
            row = db.execute(
                "SELECT role, salt, password_hash, iterations FROM accounts WHERE login = ?",
                (login,),
            ).fetchone()
        if row:
            role, salt, expected, iterations = row
            if hmac.compare_digest(self._hash(password, salt, iterations), expected):
                return Account(login, role)
        raise ValueError("Неверный логин или пароль.")

    def create(self, login: str, password: str, role: str, actor: Account | None) -> Account:
        login = login.strip().casefold()
        if not 3 <= len(login) <= 40 or any(c.isspace() for c in login):
            raise ValueError("Укажите логин длиной от 3 до 40 символов без пробелов.")
        if len(password) < 8:
            raise ValueError("Пароль должен содержать минимум 8 символов.")
        if role not in ROLES:
            raise ValueError("Выберите роль.")
        if actor is not None:
            actor.require_admin()
        salt = secrets.token_bytes(16)
        digest = self._hash(password, salt, ITERATIONS)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if actor is None and (role != ADMIN or db.execute(
                "SELECT 1 FROM accounts LIMIT 1"
            ).fetchone()):
                raise PermissionError("Первичная настройка уже выполнена. Войдите в приложение.")
            try:
                db.execute(
                    "INSERT INTO accounts VALUES (?, ?, ?, ?, ?)",
                    (login, role, salt, digest, ITERATIONS),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Этот логин уже занят.") from exc
        return Account(login, role)

    def list_users(self, actor: Account) -> list[tuple[str, str]]:
        actor.require_admin()
        with self.connect() as db:
            return db.execute("SELECT login, role FROM accounts ORDER BY login").fetchall()
