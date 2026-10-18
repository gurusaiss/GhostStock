"""
server.py — GhostStock API + static dashboard server.

Wraps the existing detection engine (detector.py, evaluate.py, generate_data.py,
load_real_data.py are all untouched) behind a small JSON API, and serves the
hand-built dashboard in static/.

Run:
    python server.py
Then open http://localhost:8000
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

app = FastAPI(title="GhostStock API")

# ----------------------------------------------------------------------------
# Dataset state — loaded once at startup; an uploaded CSV lives alongside it
# and either can be selected per-request via ?source=synthetic|upload.
# This mirrors a single local demo user, not a multi-tenant service.
# ----------------------------------------------------------------------------

_state = {
    "synthetic": None,   # (sales_df, stores_df, sku_master_df)
    "upload": None,      # (sales_df, filename)
}


def _load_synthetic_once():
    if _state["synthetic"] is None:
        _state["synthetic"] = load_synthetic()
    return _state["synthetic"]


def _get_dataset(source: str):
    if source == "upload":
        if _state["upload"] is None:
            raise HTTPException(400, "No CSV has been uploaded yet.")
        return _state["upload"][0]
    try:
        sales, _, _ = _load_synthetic_once()
    except FileNotFoundError:
        raise HTTPException(
            500, "Synthetic dataset not found — run `python generate_data.py` first."
        )
    return sales


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
# API
# ----------------------------------------------------------------------------


@app.get("/api/meta")
def get_meta(source: str = "synthetic"):
    sales = _get_dataset(source)
    stores = sorted(sales["store_id"].unique().tolist())
    dates = sales["date"].drop_duplicates().sort_values()
    min_eval_idx = max(1, int(len(dates) * 0.25))  # needs lookback history for a baseline
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
        "uploadFilename": _state["upload"][1] if (source == "upload" and _state["upload"]) else None,
    }


@app.get("/api/alerts")
def get_alerts(store: str, date: str, source: str = "synthetic"):
    sales = _get_dataset(source)
    alerts = top_alerts_for_store_day(sales, store, date, top_n=TOP_N)
    if alerts.empty:
        return {"store": store, "date": date, "alerts": []}
    return {"store": store, "date": date, "alerts": _df_to_native(alerts)}


@app.get("/api/chart")
def get_chart(store: str, sku: str, date: str, source: str = "synthetic", days: int = 60):
    sales = _get_dataset(source)
    as_of = pd.Timestamp(date)
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
def get_summary(source: str = "synthetic"):
    if source != "synthetic" or not EVAL_SUMMARY_PATH.exists():
        return {"available": False}
    with open(EVAL_SUMMARY_PATH) as f:
        summary = json.load(f)
    summary["available"] = True
    summary["precomputed"] = True
    return summary


@app.post("/api/summary/recompute")
def recompute_summary():
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
async def upload_csv(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
        df = load_real_csv(io.BytesIO(raw_bytes))
        df = normalize_uploaded(df)
    except Exception as e:
        raise HTTPException(400, f"Could not parse this file: {e}")
    if df.empty:
        raise HTTPException(400, "No usable rows found in this file after cleaning.")
    _state["upload"] = (df, file.filename)
    return {
        "filename": file.filename,
        "nRows": int(len(df)),
        "nStores": int(df["store_id"].nunique()),
        "nSkus": int(df["sku_id"].nunique()),
    }


# ----------------------------------------------------------------------------
# Static dashboard
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
