"""
detector.py — Phantom Stock-Out detection engine.

Core idea: a SKU that is genuinely in stock (system shows qty_on_hand > 0) but sells
ZERO units for an unusually long streak — relative to that SKU's OWN historical
velocity, and with no promo/holiday explaining the lull — is very likely a
"phantom" record: the physical stock is gone (lost / stolen / misplaced / miscounted)
even though the system still thinks it's there.

Method (plain statistics + a light scikit-learn ensemble boost, no deep learning):

  1. Streak detection: consecutive zero-sales days while system stock > 0, computed
     per SKU-store via vectorized pandas groupby (no slow python loops).
  2. Baseline: each SKU-store's own historical average daily sales on days it was
     actually selling (its personal velocity / seasonality baseline) — a fast
     mover needs only a few zero days to look suspicious, a slow mover needs many.
  3. Poisson implausibility score: under that baseline rate, what is the chance of
     seeing `streak_len` consecutive zero-sale days by pure luck? We flag when that
     chance is tiny.  confidence ~= 1 - exp(-baseline_rate * streak_len)
  4. Promo/holiday suppression: if a promotion was active during much of the
     streak, that explains a sales dip differently (or the opposite — a promo
     usually INCREASES expected sales, making a zero-sales promo period even more
     suspicious) — we down-weight streaks that overlap promos because the demand
     signal itself becomes noisier and harder to trust.
  5. Stock-magnitude sanity: a streak against a trivially small shown stock (e.g.
     "1 unit left") is less suspicious than the same streak against a comfortable
     shown stock (e.g. "30 units left") — the latter is a bigger inventory-record
     gap and a bigger ₹ at risk.
  6. IsolationForest cross-check: an unsupervised anomaly-detection pass over all
     SKUs in the store-day snapshot, blended in as a secondary signal so the
     ranking isn't purely one hand-tuned formula.

Hard budget: never more than `TOP_N` alerts per store-day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

TOP_N = 20
MIN_STREAK_DAYS = 5          # shorter gaps are routine noise, not worth a manager's time
MIN_SELLING_DAYS = 3         # need at least a few real selling days to trust the baseline —
                              # otherwise a SKU that's simply never been stocked/sold looks
                              # identical to a genuine phantom (no velocity to compare against)
DEFAULT_LOOKBACK_DAYS = 120  # how much history feeds the baseline/streak calc


def _streak_and_baseline(window: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per (store_id, sku_id) computation over a date-sorted window.

    Returns one row per (store_id, sku_id) present in `window`, with the streak
    of zero-sales-while-in-stock days ending at the LAST date in the window, plus
    each pair's own historical baseline sales rate.
    """
    df = window.sort_values(["store_id", "sku_id", "date"]).copy()
    key = ["store_id", "sku_id"]

    df["zero_day"] = (df["qty_sold"] == 0) & (df["qty_on_hand"] > 0)

    # --- contiguous-run id, computed without any python-level loop ---
    shifted = df.groupby(key, sort=False)["zero_day"].shift(fill_value=False)
    change = df["zero_day"] != shifted
    df["run_id"] = change.cumsum()

    run_len = df.groupby(key + ["run_id"], sort=False)["zero_day"].transform("size")
    run_promo_frac = df.groupby(key + ["run_id"], sort=False)["promo_active"].transform("mean")

    df["run_len"] = np.where(df["zero_day"], run_len, 0)
    df["run_promo_frac"] = np.where(df["zero_day"], run_promo_frac, 0.0)

    # baseline: mean/std sales on days the item was actually selling (in stock, qty_sold>0)
    selling_mask = (df["qty_on_hand"] > 0) & (df["qty_sold"] > 0)
    base = df[selling_mask].groupby(key, sort=False)["qty_sold"].agg(
        baseline_mean="mean", baseline_std="std", selling_days="size"
    )

    # the LAST row per group = the as-of date snapshot
    last_rows = df.groupby(key, sort=False).tail(1).set_index(key)

    snap = last_rows[["date", "qty_sold", "qty_on_hand", "zero_day", "run_len", "run_promo_frac"]]
    snap = snap.join(base, how="left")
    snap["baseline_mean"] = snap["baseline_mean"].fillna(0.05)
    snap["baseline_std"] = snap["baseline_std"].fillna(snap["baseline_mean"] * 0.5)
    snap["selling_days"] = snap["selling_days"].fillna(0).astype(int)

    # recent average sold over the whole window (used as an extra ML feature)
    recent_avg = df.groupby(key, sort=False)["qty_sold"].mean()
    snap["recent_avg_sold"] = recent_avg

    snap = snap.rename(columns={"run_len": "streak_len", "run_promo_frac": "promo_fraction_in_streak",
                                 "qty_on_hand": "current_stock"})
    snap["streak_len"] = snap["streak_len"].astype(int)
    return snap.reset_index()


def _score_snapshot(snap: pd.DataFrame) -> pd.DataFrame:
    """Add confidence_score (0-100) and a plain-English reason to a snapshot df."""
    snap = snap.copy()
    eligible = (
        (snap["streak_len"] >= MIN_STREAK_DAYS)
        & (snap["current_stock"] > 0)
        & (snap["selling_days"] >= MIN_SELLING_DAYS)
    )

    baseline = snap["baseline_mean"].clip(lower=0.05)
    poisson_conf = 1 - np.exp(-baseline * snap["streak_len"])

    promo_discount = 1 - 0.85 * snap["promo_fraction_in_streak"]
    stock_factor = np.tanh(snap["current_stock"] / (baseline * 3 + 1))

    raw_score = poisson_conf * promo_discount * (0.5 + 0.5 * stock_factor)
    raw_score = np.where(eligible, raw_score, 0.0)
    snap["raw_score"] = raw_score

    # --- IsolationForest cross-check over this store-day's SKU snapshot ---
    anomaly_norm = np.zeros(len(snap))
    if len(snap) >= 15:
        features = snap[["streak_len", "baseline_mean", "current_stock", "recent_avg_sold"]].fillna(0).values
        try:
            iso = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
            iso.fit(features)
            # decision_function: higher = more normal, lower = more anomalous -> flip sign
            anomaly_raw = -iso.decision_function(features)
            rng = anomaly_raw.max() - anomaly_raw.min()
            anomaly_norm = (anomaly_raw - anomaly_raw.min()) / rng if rng > 1e-9 else np.zeros(len(snap))
        except Exception:
            anomaly_norm = np.zeros(len(snap))
    snap["anomaly_norm"] = np.where(eligible, anomaly_norm, 0.0)

    final = 0.75 * snap["raw_score"] + 0.25 * snap["anomaly_norm"]
    final = np.where(eligible, final, 0.0)
    snap["confidence_score"] = np.round(np.clip(final, 0, 1) * 100, 1)
    snap["eligible"] = eligible

    snap["reason"] = snap.apply(_build_reason, axis=1)
    return snap


def _build_reason(row) -> str:
    if not row["eligible"]:
        return "Not flagged (streak too short or stock already shows zero)."
    base = (f"{int(row['streak_len'])} zero-sales days while system shows "
            f"{int(row['current_stock'])} in stock (normal pace ~{row['baseline_mean']:.1f} units/day)")
    if row["promo_fraction_in_streak"] > 0.3:
        return base + f", though a promo was active {row['promo_fraction_in_streak']*100:.0f}% of that streak — confidence reduced."
    return base + ", and no promo is active."


def snapshot_for_store_day(sales_df: pd.DataFrame, store_id: str, as_of_date,
                            lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Full scored snapshot (all SKUs, not just top 20) for one store on one date.
    Uses only data up to and including `as_of_date` — no lookahead."""
    as_of_date = pd.Timestamp(as_of_date)
    start = as_of_date - pd.Timedelta(days=lookback_days)
    window = sales_df[
        (sales_df["store_id"] == store_id)
        & (sales_df["date"] >= start)
        & (sales_df["date"] <= as_of_date)
    ]
    if window.empty:
        return pd.DataFrame()
    snap = _streak_and_baseline(window)
    snap = _score_snapshot(snap)
    snap["store_id"] = store_id
    snap["as_of_date"] = as_of_date
    return snap


def snapshot_for_all_stores_day(sales_df: pd.DataFrame, as_of_date,
                                 lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                                 store_ids=None) -> pd.DataFrame:
    """Scored snapshot for one date, computed in a single vectorized pass across
    the given `store_ids` (all stores if None). Used by evaluate.py so it doesn't
    re-filter the whole dataset once per store."""
    as_of_date = pd.Timestamp(as_of_date)
    start = as_of_date - pd.Timedelta(days=lookback_days)
    window = sales_df[(sales_df["date"] >= start) & (sales_df["date"] <= as_of_date)]
    if store_ids is not None:
        window = window[window["store_id"].isin(store_ids)]
    if window.empty:
        return pd.DataFrame()
    snap = _streak_and_baseline(window)
    # Isolation Forest needs per-store context (so one store's slow movers don't
    # get judged against another store's fast movers) — score store by store.
    # (manual loop instead of groupby.apply: newer pandas drops the group key
    # column from the sub-frame passed to apply, which _score_snapshot needs)
    scored = pd.concat(
        [_score_snapshot(sub) for _, sub in snap.groupby("store_id", sort=False)],
        ignore_index=True,
    )
    scored["as_of_date"] = as_of_date
    return scored


def top_alerts_for_store_day(sales_df: pd.DataFrame, store_id: str, as_of_date,
                              top_n: int = TOP_N,
                              lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """The main entry point: top-N (hard budget) highest-confidence phantom
    stock-out alerts for one store on one date."""
    snap = snapshot_for_store_day(sales_df, store_id, as_of_date, lookback_days)
    if snap.empty:
        return snap
    alerts = snap[snap["eligible"]].sort_values("confidence_score", ascending=False).head(top_n)
    cols = ["sku_id", "store_id", "as_of_date", "confidence_score", "reason",
            "streak_len", "baseline_mean", "current_stock", "promo_fraction_in_streak"]
    return alerts[cols].reset_index(drop=True)
