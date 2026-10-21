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

Sessions: a random opaque token in an httponly cookie. Only a SHA-256 hash of
the token is stored (in the same SQLite file), so sessions survive a server
restart and work across multiple worker processes, and a leaked database
can't be replayed as live cookies. Each lookup joins the user's current
role/store scope, so admin edits apply immediately and removing a user ends
their sessions.
"""

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "ghoststock.db"
SESSION_TTL_SECONDS = 12 * 60 * 60  # 12 hours
PBKDF2_ITERATIONS = 200_000



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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires REAL NOT NULL
        )"""
    )
    conn.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
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


def _guard_last_admin(conn, user_id: int, new_role: str | None = None):
    """Refuses an operation that would leave the mall with zero admins — either
    deleting the last admin, or demoting them to manager."""
    row = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if not row or row["role"] != "admin":
        return
    if new_role == "admin":
        return  # staying admin — not a demotion
    n_admins = conn.execute("SELECT COUNT(*) AS c FROM users WHERE role='admin'").fetchone()["c"]
    if n_admins <= 1:
        raise ValueError("Can't remove the last remaining admin account.")


def delete_user(user_id: int):
    conn = get_db()
    try:
        _guard_last_admin(conn, user_id)
        conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def update_user_access(user_id: int, role: str, store_ids: str = "ALL"):
    """Admin editing an existing user's role / store scope (not their password)."""
    if role not in ("admin", "manager"):
        raise ValueError("role must be 'admin' or 'manager'")
    conn = get_db()
    try:
        _guard_last_admin(conn, user_id, new_role=role)
        scope = "ALL" if role == "admin" else (store_ids or "ALL")
        conn.execute("UPDATE users SET role=?, store_ids=? WHERE id=?", (role, scope, user_id))
        conn.commit()
    finally:
        conn.close()


def reset_password(user_id: int, new_password: str):
    """Admin force-setting someone else's password (e.g. they're locked out)."""
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    digest, salt = hash_password(new_password)
    conn = get_db()
    conn.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?", (digest, salt, user_id))
    conn.commit()
    conn.close()
    destroy_user_sessions(user_id)  # whoever knew the old password is signed out


def change_own_password(user_id: int, current_password: str, new_password: str, keep_token: str | None = None):
    """Self-service password change — requires knowing the current password,
    unlike admin reset_password. The right way for anyone (including the
    admin) to stop using a temporary/admin-assigned password long-term."""
    if len(new_password) < 8:
        raise ValueError("New password must be at least 8 characters.")
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("User not found.")
    digest, _ = hash_password(current_password, row["salt"])
    if not secrets.compare_digest(digest, row["password_hash"]):
        conn.close()
        raise ValueError("Current password is incorrect.")
    new_digest, new_salt = hash_password(new_password)
    conn.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?", (new_digest, new_salt, user_id))
    if keep_token:
        conn.execute("DELETE FROM sessions WHERE user_id=? AND token_hash != ?",
                     (user_id, _token_hash(keep_token)))
    else:
        conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- login throttle
# A minimal brute-force guard: N failed attempts for an email locks it out for
# a short window. In-memory, so it resets on restart — fine for the threat
# model here (a local/LAN tool), not a substitute for a real rate limiter if
# this is ever put on the open internet.

_FAILED_LOGINS: dict = {}  # email -> {"count": int, "locked_until": float}
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def login_locked_out(email: str) -> float:
    """Returns seconds remaining if locked out, else 0."""
    rec = _FAILED_LOGINS.get(email.strip().lower())
    if not rec:
        return 0
    remaining = rec["locked_until"] - time.time()
    return max(0, remaining)


def _record_login_failure(email: str):
    email = email.strip().lower()
    rec = _FAILED_LOGINS.setdefault(email, {"count": 0, "locked_until": 0})
    rec["count"] += 1
    if rec["count"] >= MAX_LOGIN_ATTEMPTS:
        rec["locked_until"] = time.time() + LOCKOUT_SECONDS
        rec["count"] = 0


def _clear_login_failures(email: str):
    _FAILED_LOGINS.pop(email.strip().lower(), None)


def verify_login(email: str, password: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    conn.close()
    if not row:
        _record_login_failure(email)
        return None
    digest, _ = hash_password(password, row["salt"])
    if not secrets.compare_digest(digest, row["password_hash"]):
        _record_login_failure(email)
        return None
    _clear_login_failures(email)
    return dict(row)


def _token_hash(token: str) -> str:
    # only a hash of the token is stored, so a leaked DB can't be replayed as cookies
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user: dict) -> str:
    token = secrets.token_urlsafe(32)
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, expires) VALUES (?, ?, ?)",
        (_token_hash(token), user["id"], time.time() + SESSION_TTL_SECONDS),
    )
    conn.commit()
    conn.close()
    return token


def get_session(token: str | None):
    """Looks the session up together with the user's CURRENT role/scope, so an
    admin's edit takes effect on the next request and a removed user's
    session dies with the user (ON DELETE CASCADE)."""
    if not token:
        return None
    conn = get_db()
    conn.execute("PRAGMA foreign_keys = ON")
    row = conn.execute(
        """SELECT s.expires, u.id AS user_id, u.email, u.role, u.store_ids
           FROM sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token_hash = ?""",
        (_token_hash(token),),
    ).fetchone()
    if row and row["expires"] < time.time():
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
        row = None
    conn.close()
    return dict(row) if row else None


def destroy_session(token: str | None):
    if token:
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
        conn.close()


def destroy_user_sessions(user_id: int):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def allowed_stores(user: dict, all_stores: list[str]) -> list[str]:
    if user["store_ids"] == "ALL":
        return all_stores
    allowed = {s.strip() for s in user["store_ids"].split(",") if s.strip()}
    return [s for s in all_stores if s in allowed]
