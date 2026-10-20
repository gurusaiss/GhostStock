"""
auth.py — Admin-provisioned accounts for GhostStock.

Design decision: there is NO public self-signup. A mall has one admin account
(the platform owner / IT contact); that admin creates a login (email +
password) for every store manager who needs access, and optionally restricts
each one to the stores they're responsible for. This matches how an internal
retail-ops tool should actually be rolled out — nobody should be able to
register themselves into a mall's live inventory data, and a manager should
only see the store(s) they're accountable for, not the whole chain.

Storage: a local SQLite file (data/ghoststock.db) — no external auth service,
no email delivery required, fully offline. Passwords are never stored in
plain text: PBKDF2-HMAC-SHA256 with a random per-user salt (stdlib only, no
extra dependency).

Sessions: a random opaque token in an httponly cookie, mapped server-side to
the session record. Sessions live in memory, so restarting the server logs
everyone out — acceptable for a local/single-process demo; swap SESSIONS for
a persisted store (e.g. a table, or Redis) before running this multi-process
or behind a reverse proxy with multiple workers.
"""

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "ghoststock.db"
SESSION_TTL_SECONDS = 12 * 60 * 60  # 12 hours
PBKDF2_ITERATIONS = 200_000

# token -> {user_id, email, role, store_ids, expires}
SESSIONS: dict = {}


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt_hex = salt_hex or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    ).hex()
    return digest, salt_hex


def init_db():
    """Creates the users table if missing, and seeds a default admin account
    the very first time the app runs (so there's always a way in). Returns
    (email, password) of the seeded admin if one was just created, else None.
    """
    conn = get_db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','manager')),
            store_ids TEXT NOT NULL DEFAULT 'ALL',
            created_at REAL NOT NULL
        )"""
    )
    conn.commit()
    n_users = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]

    seeded = None
    if n_users == 0:
        import os

        email = os.environ.get("GHOSTSTOCK_ADMIN_EMAIL", "admin@ghoststock.local")
        password = os.environ.get("GHOSTSTOCK_ADMIN_PASSWORD") or secrets.token_urlsafe(9)
        _insert_user(conn, email, password, "admin", "ALL")
        seeded = (email, password)
    conn.close()
    return seeded


def _insert_user(conn, email: str, password: str, role: str, store_ids: str):
    digest, salt = hash_password(password)
    conn.execute(
        "INSERT INTO users (email, password_hash, salt, role, store_ids, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (email.strip().lower(), digest, salt, role, store_ids, time.time()),
    )
    conn.commit()


def create_user(email: str, password: str, role: str, store_ids: str = "ALL"):
    if role not in ("admin", "manager"):
        raise ValueError("role must be 'admin' or 'manager'")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    conn = get_db()
    try:
        _insert_user(conn, email, password, role, store_ids or "ALL")
    finally:
        conn.close()


def list_users():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, email, role, store_ids, created_at FROM users ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user(user_id: int):
    conn = get_db()
    # never allow deleting the last remaining admin — the mall would lock itself out
    row = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if row and row["role"] == "admin":
        n_admins = conn.execute("SELECT COUNT(*) AS c FROM users WHERE role='admin'").fetchone()["c"]
        if n_admins <= 1:
            conn.close()
            raise ValueError("Can't delete the last remaining admin account.")
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def reset_password(user_id: int, new_password: str):
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    digest, salt = hash_password(new_password)
    conn = get_db()
    conn.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?", (digest, salt, user_id))
    conn.commit()
    conn.close()


def verify_login(email: str, password: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    conn.close()
    if not row:
        return None
    digest, _ = hash_password(password, row["salt"])
    if not secrets.compare_digest(digest, row["password_hash"]):
        return None
    return dict(row)


def create_session(user: dict) -> str:
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {
        "user_id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "store_ids": user["store_ids"],
        "expires": time.time() + SESSION_TTL_SECONDS,
    }
    return token


def get_session(token: str | None):
    if not token:
        return None
    sess = SESSIONS.get(token)
    if not sess:
        return None
    if sess["expires"] < time.time():
        SESSIONS.pop(token, None)
        return None
    return sess


def destroy_session(token: str | None):
    if token:
        SESSIONS.pop(token, None)


def allowed_stores(user: dict, all_stores: list[str]) -> list[str]:
    if user["store_ids"] == "ALL":
        return all_stores
    allowed = {s.strip() for s in user["store_ids"].split(",") if s.strip()}
    return [s for s in all_stores if s in allowed]
