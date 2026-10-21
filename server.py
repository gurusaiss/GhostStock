"""
server.py — GhostStock API + static dashboard server.

Wraps the existing detection engine (detector.py, evaluate.py, generate_data.py,
load_real_data.py are all untouched) behind a small JSON API, and serves the
hand-built dashboard in static/.

Access model: there is no public signup. The mall's admin account (seeded on
first run — see the console output) creates a login for every store manager
who needs access, optionally scoped to just the store(s) they run. See
auth.py for the full reasoning.

Run:
    python server.py
Then open http://localhost:8000 — the console prints the default admin
login the very first time the database is created.
"""

import io
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth
from data_io import load_synthetic, load_ground_truth, normalize_uploaded
from load_real_data import load_real_csv
from detector import (
    top_alerts_for_store_day,
    snapshot_for_store_day,
    snapshot_for_all_stores_day,
    TOP_N,
)

APP_DIR = Path(__file__).parent
EVAL_SUMMARY_PATH = APP_DIR / "data" / "eval_summary.json"
SESSION_COOKIE = "gs_session"
# Set GHOSTSTOCK_SECURE_COOKIES=1 when serving over HTTPS so the session cookie
# is never sent in the clear. Off by default so plain http://localhost works.
SECURE_COOKIES = os.environ.get("GHOSTSTOCK_SECURE_COOKIES") == "1"

SEEDED_ADMIN = None


@asynccontextmanager
async def lifespan(_app):
    """Create the users table (and seed the first admin) however the app is
    launched — `python server.py` or `uvicorn server:app`."""
    global SEEDED_ADMIN
    SEEDED_ADMIN = auth.init_db()
    if SEEDED_ADMIN:
        email, password = SEEDED_ADMIN
        print("=" * 60)
        print("First run — an admin account was created:")
        print(f"  Email:    {email}")
        print(f"  Password: {password}")
        print("Sign in, then add mall staff from the Users tab. Set")
        print("GHOSTSTOCK_ADMIN_EMAIL / GHOSTSTOCK_ADMIN_PASSWORD to pin these.")
        print("=" * 60)
    yield


app = FastAPI(title="GhostStock API", lifespan=lifespan)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@app.middleware("http")
async def require_ajax_header(request: Request, call_next):
    """CSRF defence for the cookie session: every state-changing /api call must
    carry a custom header. A cross-site <form> or <img> can't set one, and a
    cross-site fetch that does triggers a CORS preflight this server never
    approves. SameSite=Lax on the cookie is the second layer."""
    if (request.url.path.startswith("/api/")
            and request.method in ("POST", "PUT", "PATCH", "DELETE")
            and request.headers.get("x-requested-with") != "ghoststock"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Missing CSRF header."}, status_code=403)
    return await call_next(request)

# ----------------------------------------------------------------------------
# Dataset state — the synthetic set loads once at startup; an uploaded CSV is
# kept per logged-in user (keyed by user id) so two managers uploading
# different files don't clobber each other.
# ----------------------------------------------------------------------------

_state = {
    "synthetic": None,     # (sales_df, stores_df, sku_master_df)
    "uploads": {},          # user_id -> (sales_df, filename)
}


def _load_synthetic_once():
    if _state["synthetic"] is None:
        _state["synthetic"] = load_synthetic()
    return _state["synthetic"]


def _get_dataset(source: str, user: dict):
    if source == "upload":
        entry = _state["uploads"].get(user["user_id"])
        if entry is None:
            raise HTTPException(400, "No CSV has been uploaded yet.")
        return entry[0]
    try:
        sales, _, _ = _load_synthetic_once()
    except FileNotFoundError:
        raise HTTPException(
            500, "Synthetic dataset not found — run `python generate_data.py` first."
        )
    return sales


def _parse_date(date: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(date)
    except (ValueError, TypeError):
        raise HTTPException(400, f"Invalid date: {date!r} (expected YYYY-MM-DD).")


def _df_to_native(df: pd.DataFrame) -> list:
    """Row-oriented records with numpy/pandas scalars converted to plain
    Python types so FastAPI's JSON encoder doesn't choke on them."""
    out = df.replace({np.nan: None}).to_dict(orient="records")
    for row in out:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
            elif isinstance(v, pd.Timestamp):
                row[k] = v.strftime("%Y-%m-%d")
    return out


# ----------------------------------------------------------------------------
# Auth dependencies & endpoints
# ----------------------------------------------------------------------------


def current_user(gs_session: Optional[str] = Cookie(default=None)) -> dict:
    sess = auth.get_session(gs_session)
    if not sess:
        raise HTTPException(401, "Not authenticated")
    return sess


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    return user


def _store_scope(user: dict, all_stores: list, source: str = "synthetic") -> list:
    """Stores this user may see. Scoping applies to the shared dataset; a CSV the
    user uploaded themselves is private to them, so all its stores are theirs."""
    if source == "upload":
        return all_stores
    return auth.allowed_stores(user, all_stores)


def _assert_store_allowed(user: dict, store: str, all_stores: list, source: str = "synthetic"):
    if store not in _store_scope(user, all_stores, source):
        raise HTTPException(403, f"You don't have access to store {store}.")


class LoginPayload(BaseModel):
    email: str
    password: str


class NewUserPayload(BaseModel):
    email: str
    password: str
    role: str  # "admin" | "manager"
    store_ids: str = "ALL"  # "ALL" or comma-separated store ids


class ResetPasswordPayload(BaseModel):
    password: str


class EditUserPayload(BaseModel):
    role: str
    store_ids: str = "ALL"


class ChangePasswordPayload(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/auth/login")
def login(payload: LoginPayload, response: Response):
    wait = auth.login_locked_out(payload.email)
    if wait:
        raise HTTPException(429, f"Too many failed attempts. Try again in {int(wait) + 1}s.")
    user = auth.verify_login(payload.email, payload.password)
    if not user:
        raise HTTPException(401, "Incorrect email or password.")
    token = auth.create_session(user)
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="lax", secure=SECURE_COOKIES,
        max_age=auth.SESSION_TTL_SECONDS,
    )
    return {"email": user["email"], "role": user["role"]}


@app.post("/api/auth/logout")
def logout(response: Response, gs_session: Optional[str] = Cookie(default=None)):
    auth.destroy_session(gs_session)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return {"email": user["email"], "role": user["role"]}


@app.post("/api/auth/change_password")
def change_my_password(payload: ChangePasswordPayload, user: dict = Depends(current_user),
                       gs_session: Optional[str] = Cookie(default=None)):
    """Anyone — including the admin — can swap a temporary/admin-assigned
    password for one only they know. Requires the current password, unlike
    an admin's force-reset of someone else's account."""
    try:
        auth.change_own_password(user["user_id"], payload.current_password, payload.new_password,
                                 keep_token=gs_session)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


# ----------------------------------------------------------------------------
# Admin: user management (the only way new logins get created — no signup)
# ----------------------------------------------------------------------------


@app.get("/api/admin/users")
def admin_list_users(admin: dict = Depends(require_admin)):
    return auth.list_users()


@app.post("/api/admin/users")
def admin_create_user(payload: NewUserPayload, admin: dict = Depends(require_admin)):
    if "@" not in payload.email or " " in payload.email.strip():
        raise HTTPException(400, "Enter a valid email address.")
    try:
        auth.create_user(payload.email, payload.password, payload.role, payload.store_ids)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(400, "A user with that email already exists.")
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.patch("/api/admin/users/{user_id}")
def admin_edit_user(user_id: int, payload: EditUserPayload, admin: dict = Depends(require_admin)):
    """Change an existing user's role / store scope — e.g. a manager gets
    moved to a different store. Does not touch their password."""
    try:
        auth.update_user_access(user_id, payload.role, payload.store_ids)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/admin/users/{user_id}/reset_password")
def admin_reset_password(user_id: int, payload: ResetPasswordPayload, admin: dict = Depends(require_admin)):
    try:
        auth.reset_password(user_id, payload.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: int, admin: dict = Depends(require_admin)):
    try:
        auth.delete_user(user_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/admin/generate_password")
def admin_generate_password(admin: dict = Depends(require_admin)):
    return {"password": secrets.token_urlsafe(9)}


# ----------------------------------------------------------------------------
# Data API (all gated behind a valid session; store access scoped by role)
# ----------------------------------------------------------------------------


@app.get("/api/meta")
def get_meta(source: str = "synthetic", user: dict = Depends(current_user)):
    sales = _get_dataset(source, user)
    all_stores = sorted(sales["store_id"].unique().tolist())
    stores = _store_scope(user, all_stores, source)
    dates = sales["date"].drop_duplicates().sort_values()
    min_eval_idx = max(1, int(len(dates) * 0.25))  # needs lookback history for a baseline
    entry = _state["uploads"].get(user["user_id"])
    return {
        "source": source,
        "stores": stores,
        "nRows": int(len(sales)),
        "nStores": int(sales["store_id"].nunique()),
        "nSkus": int(sales["sku_id"].nunique()),
        "dateMin": dates.iloc[0].strftime("%Y-%m-%d"),
        "dateMax": dates.iloc[-1].strftime("%Y-%m-%d"),
        "firstUsableDate": dates.iloc[min_eval_idx].strftime("%Y-%m-%d"),
        "allDates": [d.strftime("%Y-%m-%d") for d in dates.iloc[min_eval_idx:]],
        "uploadFilename": entry[1] if (source == "upload" and entry) else None,
    }


@app.get("/api/alerts")
def get_alerts(store: str, date: str, source: str = "synthetic", user: dict = Depends(current_user)):
    sales = _get_dataset(source, user)
    _assert_store_allowed(user, store, sorted(sales["store_id"].unique().tolist()), source)
    _parse_date(date)
    alerts = top_alerts_for_store_day(sales, store, date, top_n=TOP_N)
    if alerts.empty:
        return {"store": store, "date": date, "alerts": []}
    return {"store": store, "date": date, "alerts": _df_to_native(alerts)}


@app.get("/api/chart")
def get_chart(store: str, sku: str, date: str, source: str = "synthetic", days: int = 60,
              user: dict = Depends(current_user)):
    sales = _get_dataset(source, user)
    _assert_store_allowed(user, store, sorted(sales["store_id"].unique().tolist()), source)
    as_of = _parse_date(date)
    days = max(1, min(days, 365))
    hist = sales[
        (sales["store_id"] == store)
        & (sales["sku_id"] == sku)
        & (sales["date"] <= as_of)
        & (sales["date"] > as_of - pd.Timedelta(days=days))
    ].sort_values("date")
    return {
        "sku": sku,
        "store": store,
        "dates": [d.strftime("%Y-%m-%d") for d in hist["date"]],
        "sold": [int(x) for x in hist["qty_sold"]],
        "stock": [int(x) for x in hist["qty_on_hand"]],
        "promoActive": [int(x) for x in hist["promo_active"]],
    }


@app.get("/api/summary")
def get_summary(source: str = "synthetic", user: dict = Depends(current_user)):
    if source != "synthetic" or not EVAL_SUMMARY_PATH.exists():
        return {"available": False}
    with open(EVAL_SUMMARY_PATH) as f:
        summary = json.load(f)
    summary["available"] = True
    summary["precomputed"] = True
    return summary


@app.post("/api/summary/recompute")
def recompute_summary(user: dict = Depends(current_user)):
    """Live re-run of the same scan evaluate.py does, across every ground-truth
    phantom case's end date. Slow (~3-4 min on the full synthetic dataset) —
    the frontend warns about this before calling it."""
    sales, stores, sku_master = _load_synthetic_once()
    gt = load_ground_truth()
    sku_price = sku_master.set_index("sku_id")["unit_price"].to_dict()

    caught_rows = []
    for date, group in gt.groupby("phantom_end_date"):
        needed_stores = group["store_id"].unique().tolist()
        snapshot = snapshot_for_all_stores_day(sales, date, store_ids=needed_stores)
        if snapshot.empty:
            continue
        for store_id in needed_stores:
            store_snap = snapshot[snapshot["store_id"] == store_id]
            top = store_snap[store_snap["eligible"]].sort_values(
                "confidence_score", ascending=False
            ).head(TOP_N)
            flagged = set(top["sku_id"]) if not top.empty else set()
            sub = group[group["store_id"] == store_id]
            for _, case in sub.iterrows():
                caught_rows.append({**case.to_dict(), "caught": case["sku_id"] in flagged})

    caught_df = pd.DataFrame(caught_rows)
    confirmed = caught_df[caught_df["caught"]].copy()
    confirmed["unit_price"] = confirmed["sku_id"].map(sku_price)
    confirmed["rupees_lost"] = (confirmed["pre_phantom_avg_daily_sales"]
                                 * confirmed["unit_price"] * confirmed["phantom_length_days"])

    n_confirmed = len(confirmed)
    total_rupees_window = float(confirmed["rupees_lost"].sum())
    n_days_in_data = (sales["date"].max() - sales["date"].min()).days + 1
    years_in_data = n_days_in_data / 365.0
    annualized = total_rupees_window / years_in_data if years_in_data > 0 else 0

    return {
        "available": True,
        "precomputed": False,
        "n_confirmed": int(n_confirmed),
        "n_ground_truth_cases": int(len(gt)),
        "total_rupees_window": total_rupees_window,
        "n_days_in_data": int(n_days_in_data),
        "years_in_data": float(years_in_data),
        "annualized_rupees": float(annualized),
        "n_stores": int(sales["store_id"].nunique()),
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...), user: dict = Depends(current_user)):
    try:
        raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than the 50 MB upload limit.")
        df = load_real_csv(io.BytesIO(raw_bytes))
        df = normalize_uploaded(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse this file: {e}")
    if df.empty:
        raise HTTPException(400, "No usable rows found in this file after cleaning.")
    _state["uploads"][user["user_id"]] = (df, file.filename)
    return {
        "filename": file.filename,
        "nRows": int(len(df)),
        "nStores": int(df["store_id"].nunique()),
        "nSkus": int(df["sku_id"].nunique()),
    }


# ----------------------------------------------------------------------------
# Static dashboard — index.html itself is public (it's an inert shell without
# a session; every data call behind it is gated above), login.html is always
# public.
# ----------------------------------------------------------------------------

STATIC_DIR = APP_DIR / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    print("GhostStock running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
