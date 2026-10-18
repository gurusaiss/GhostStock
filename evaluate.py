"""
evaluate.py — Score the detector against the HIDDEN ground truth labels.

The detector itself never sees ground_truth.csv — this script only uses it
afterwards to check how well the detector's rankings line up with the phantom
stock-outs we deliberately injected into the synthetic data.

Computes:
  - Precision@20 per store-day (of the top-20 alerts, how many were real phantom
    stock-outs that were actually active that day)
  - Recall across all injected phantom stock-out cases (of all the phantom cases
    we planted, what fraction got caught in the store's top-20 by the time the
    streak had fully played out)
  - An estimated annual Rs recovered, with the arithmetic printed step by step

Run:
  python evaluate.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_synthetic, load_ground_truth
from detector import snapshot_for_all_stores_day, TOP_N

N_SAMPLE_RANDOM_DAYS = 15  # extra "ordinary day" samples, for a realistic precision read
RNG_SEED = 7
SUMMARY_PATH = Path(__file__).parent / "data" / "eval_summary.json"


def main():
    t_start = time.time()
    print("=" * 70)
    print("GHOSTSTOCK - EVALUATION AGAINST HIDDEN GROUND TRUTH")
    print("=" * 70)

    sales, stores, sku_master = load_synthetic()
    gt = load_ground_truth()
    print(f"\nLoaded {len(sales):,} sales/inventory rows, {len(gt)} injected phantom "
          f"stock-out cases (hidden ground truth, eval-only).")

    rng = np.random.default_rng(RNG_SEED)

    # ---- Evaluation days: each phantom case's own END date (the day the streak
    #      has fully played out) + a handful of random "ordinary" store-days, for
    #      a realistic precision read that isn't cherry-picked to phantom cases ----
    gt_days = gt[["store_id", "phantom_end_date"]].drop_duplicates()
    all_store_ids = stores["store_id"].tolist()
    all_dates = sales["date"].drop_duplicates().sort_values()
    min_eval_date = all_dates.iloc[int(len(all_dates) * 0.3)]  # need history for baseline
    candidate_dates = all_dates[all_dates >= min_eval_date]
    random_days = pd.DataFrame({
        "store_id": rng.choice(all_store_ids, size=N_SAMPLE_RANDOM_DAYS),
        "phantom_end_date": rng.choice(candidate_dates, size=N_SAMPLE_RANDOM_DAYS),
    })
    eval_days = pd.concat([gt_days, random_days], ignore_index=True).drop_duplicates()

    n_unique_dates = eval_days["phantom_end_date"].nunique()
    print(f"Evaluating detector on {len(eval_days)} store-days across {n_unique_dates} "
          f"unique dates ({len(gt_days)} phantom-case end-dates + "
          f"{N_SAMPLE_RANDOM_DAYS} random ordinary days)...\n")

    # gt lookup indexed by (store,date) -> set of skus with an ACTIVE phantom that day
    gt_by_store_date_active = {}
    for _, row in gt.iterrows():
        for d in pd.date_range(row["phantom_start_date"], row["phantom_end_date"]):
            gt_by_store_date_active.setdefault((row["store_id"], d), set()).add(row["sku_id"])

    # case -> was it caught in its store's top-20 on its own end date?
    case_caught = {}  # (store_id, sku_id, phantom_end_date) -> bool

    total_tp, total_alerts = 0, 0
    per_day_precisions = []
    n_dates_done = 0

    # One pass per unique date: compute the snapshot ONLY for the stores actually
    # needed that day (not all 25), reuse it for both precision and recall.
    for date, group in eval_days.groupby("phantom_end_date"):
        needed_stores = group["store_id"].unique().tolist()
        snapshot = snapshot_for_all_stores_day(sales, date, store_ids=needed_stores)
        if snapshot.empty:
            n_dates_done += 1
            continue

        for store_id in needed_stores:
            store_snap = snapshot[snapshot["store_id"] == store_id]
            top = store_snap[store_snap["eligible"]].sort_values(
                "confidence_score", ascending=False
            ).head(TOP_N)
            if top.empty:
                continue

            active_sku_set = gt_by_store_date_active.get((store_id, pd.Timestamp(date)), set())
            n_tp = int(top["sku_id"].isin(active_sku_set).sum())
            total_tp += n_tp
            total_alerts += len(top)
            per_day_precisions.append(n_tp / len(top))

            # record recall outcome for any ground-truth case whose own end date is
            # exactly this (store, date) pair
            this_case_mask = (gt["store_id"] == store_id) & (gt["phantom_end_date"] == date)
            for sku_id in gt.loc[this_case_mask, "sku_id"]:
                case_caught[(store_id, sku_id, date)] = sku_id in set(top["sku_id"])

        n_dates_done += 1
        if n_dates_done % 50 == 0:
            print(f"  ... {n_dates_done}/{n_unique_dates} dates processed "
                  f"({time.time()-t_start:.0f}s elapsed)")

    mean_precision = float(np.mean(per_day_precisions)) if per_day_precisions else 0.0
    print(f"\nPRECISION@{TOP_N}")
    print(f"  Across {len(per_day_precisions)} store-days evaluated:")
    print(f"  total true-phantom alerts caught = {total_tp}")
    print(f"  total alerts issued              = {total_alerts}")
    print(f"  mean precision@{TOP_N} (per store-day avg) = {mean_precision:.1%}")

    # ---- Recall ----
    gt["caught"] = [
        case_caught.get((r["store_id"], r["sku_id"], r["phantom_end_date"]), False)
        for _, r in gt.iterrows()
    ]
    caught = int(gt["caught"].sum())
    recall = caught / len(gt) if len(gt) else 0.0
    print(f"\nRECALL (on hidden phantom stock-out labels)")
    print(f"  Injected phantom stock-out cases : {len(gt)}")
    print(f"  Caught in that store's top-{TOP_N} by end of streak : {caught}")
    print(f"  Recall = {caught}/{len(gt)} = {recall:.1%}")

    # ---- Estimated annual rupee recovered ----
    print(f"\n{'='*70}")
    print("ESTIMATED ANNUAL RUPEE RECOVERED")
    print(f"{'='*70}")
    sku_price = sku_master.set_index("sku_id")["unit_price"].to_dict()

    # Use the confirmed (caught) phantom cases as our "confirmed/likely" set --
    # exactly what a manager would see validated after walking the aisle.
    confirmed = gt[gt["caught"]].copy()
    confirmed["unit_price"] = confirmed["sku_id"].map(sku_price)

    print(f"\nFormula per confirmed case:")
    print(f"  recovered_value = avg_daily_sales_when_in_stock x unit_price x days_lost\n")

    confirmed["rupees_lost_per_case"] = (
        confirmed["pre_phantom_avg_daily_sales"] * confirmed["unit_price"] * confirmed["phantom_length_days"]
    )

    n_confirmed = len(confirmed)
    total_days_lost = confirmed["phantom_length_days"].sum()
    avg_daily_sales = confirmed["pre_phantom_avg_daily_sales"].mean() if n_confirmed else 0
    avg_price = confirmed["unit_price"].mean() if n_confirmed else 0
    total_rupees_in_sample_window = confirmed["rupees_lost_per_case"].sum()

    print(f"  Confirmed/likely phantom cases (caught by detector) : {n_confirmed}")
    print(f"  Avg daily sales when in stock (across confirmed)    : {avg_daily_sales:.2f} units/day")
    print(f"  Avg unit price (across confirmed)                   : Rs {avg_price:,.2f}")
    print(f"  Total days lost across confirmed cases               : {total_days_lost} days")
    print(f"  Sum over cases of (avg_daily_sales x unit_price x days_lost):")
    print(f"      = Rs {total_rupees_in_sample_window:,.0f}  "
          f"(this is the value lost DURING the ~18-month observed window)")

    n_days_in_data = (sales["date"].max() - sales["date"].min()).days + 1
    years_in_data = n_days_in_data / 365.0
    annualized = total_rupees_in_sample_window / years_in_data if years_in_data > 0 else 0

    print(f"\n  Observed window span : {n_days_in_data} days = {years_in_data:.2f} years")
    print(f"  Annualized estimate  = Rs {total_rupees_in_sample_window:,.0f} / {years_in_data:.2f} years")
    print(f"                       = Rs {annualized:,.0f} per year, across all "
          f"{sales['store_id'].nunique()} stores")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Precision@{TOP_N} : {mean_precision:.1%}")
    print(f"  Recall            : {recall:.1%}  ({caught}/{len(gt)} phantom cases caught)")
    print(f"  Est. Rs recovered/year : Rs {annualized:,.0f}")
    print(f"\n(Evaluation completed in {time.time()-t_start:.1f}s)")

    # ---- Persist a summary so the Streamlit app can show these numbers
    #      instantly instead of re-running this ~4-minute scan on every load. ----
    summary = {
        "precision_at_20": mean_precision,
        "recall": recall,
        "n_ground_truth_cases": int(len(gt)),
        "n_caught": int(caught),
        "n_store_days_evaluated": len(per_day_precisions),
        "total_tp_alerts": int(total_tp),
        "total_alerts_issued": int(total_alerts),
        "n_confirmed": int(n_confirmed),
        "avg_daily_sales_confirmed": float(avg_daily_sales),
        "avg_unit_price_confirmed": float(avg_price),
        "total_days_lost_confirmed": int(total_days_lost),
        "total_rupees_window": float(total_rupees_in_sample_window),
        "n_days_in_data": int(n_days_in_data),
        "years_in_data": float(years_in_data),
        "annualized_rupees": float(annualized),
        "n_stores": int(sales["store_id"].nunique()),
        "generated_from_seed": RNG_SEED,
    }
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
