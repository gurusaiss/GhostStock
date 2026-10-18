"""
generate_data.py — Synthetic retail data generator for GhostStock.

Produces (in ./data):
  stores.csv              store master (store_id, size, format, city_tier)
  sku_master.csv          sku master (sku_id, category, base_demand, unit_price)
  promo_calendar.csv      promo calendar (date, sku_id, promo_name, uplift_factor)
  sales_inventory.csv     daily sales + system on-hand inventory per sku-store-day
  ground_truth.csv        HIDDEN labels of injected phantom stock-outs (eval only,
                           never fed to the detector)

Run:
  python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_SKUS = 800
N_STORES = 25
N_DAYS = 18 * 30  # ~18 months of daily history
PHANTOM_RATE = 0.03  # fraction of sku-store pairs that get a phantom stock-out injected
SKU_STOCKED_PROB = 0.85  # fraction of skus actually carried/sold in a given store

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def build_store_master(rng):
    formats = ["Hypermarket", "Supermarket", "Convenience", "Express"]
    city_tiers = ["Tier-1", "Tier-2", "Tier-3"]
    rows = []
    for i in range(N_STORES):
        fmt = rng.choice(formats, p=[0.15, 0.40, 0.30, 0.15])
        tier = rng.choice(city_tiers, p=[0.4, 0.35, 0.25])
        size_sqft = {
            "Hypermarket": rng.integers(25000, 60000),
            "Supermarket": rng.integers(8000, 25000),
            "Convenience": rng.integers(1500, 5000),
            "Express": rng.integers(800, 2000),
        }[fmt]
        rows.append({
            "store_id": f"S{i+1:03d}",
            "store_name": f"Store {i+1:03d}",
            "format": fmt,
            "city_tier": tier,
            "size_sqft": int(size_sqft),
        })
    return pd.DataFrame(rows)


def build_sku_master(rng):
    categories = [
        "Beverages", "Snacks", "Personal Care", "Home Care", "Dairy",
        "Staples", "Confectionery", "Apparel", "Electronics Accessories", "Stationery",
    ]
    rows = []
    for i in range(N_SKUS):
        cat = rng.choice(categories)
        base_demand = np.round(rng.gamma(shape=2.0, scale=3.0) + 0.3, 2)
        price = float(np.round(rng.uniform(20, 1500), 2))
        rows.append({
            "sku_id": f"SKU{i+1:04d}",
            "category": cat,
            "base_demand": base_demand,
            "unit_price": price,
        })
    return pd.DataFrame(rows)


def build_promo_calendar(rng, dates, sku_master):
    rows = []
    n_events = 70
    sku_ids = sku_master["sku_id"].values
    for _ in range(n_events):
        start_idx = rng.integers(0, len(dates) - 7)
        duration = rng.integers(2, 7)
        n_skus_in_promo = rng.integers(5, 40)
        skus = rng.choice(sku_ids, size=n_skus_in_promo, replace=False)
        promo_name = rng.choice(["Weekend Bonanza", "Festival Sale", "Clearance",
                                  "Buy1Get1", "Season Opener", "Flash Deal"])
        uplift = rng.uniform(1.4, 3.0)
        for d_off in range(duration):
            d = dates[start_idx + d_off]
            for sku in skus:
                rows.append({"date": d, "sku_id": sku, "promo_name": promo_name,
                             "uplift_factor": round(float(uplift), 2)})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(RNG_SEED)

    stores = build_store_master(rng)
    skus = build_sku_master(rng)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS, freq="D")

    promo_calendar = build_promo_calendar(rng, dates, skus)

    stores.to_csv(OUT_DIR / "stores.csv", index=False)
    skus.to_csv(OUT_DIR / "sku_master.csv", index=False)
    promo_calendar.to_csv(OUT_DIR / "promo_calendar.csv", index=False)

    store_ids = stores["store_id"].values
    sku_ids = skus["sku_id"].values
    sku_index = {s: i for i, s in enumerate(sku_ids)}
    base_demand_arr = skus["base_demand"].values.astype(float)

    size_rank = stores.set_index("store_id")["size_sqft"]
    store_factor_map = (size_rank / size_rank.median()).clip(0.3, 3.0).to_dict()

    # ---- Precompute per-date seasonal/weekday multipliers (vectorized, once) ----
    doy = dates.dayofyear.values
    season_mult_arr = np.clip(1.0 + 0.35 * np.sin((doy - 60) / 365.0 * 2 * np.pi), 0.4, None)
    wd_map = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.35, 1.3])
    weekday_mult_arr = wd_map[dates.weekday.values]
    day_base_mult_arr = season_mult_arr * weekday_mult_arr  # shape (N_DAYS,)

    # ---- Precompute promo multiplier/name/active per (sku, date) as dense arrays ----
    n_sku = len(sku_ids)
    promo_mult_mat = np.ones((n_sku, N_DAYS), dtype=float)
    promo_active_mat = np.zeros((n_sku, N_DAYS), dtype=bool)
    promo_name_mat = np.full((n_sku, N_DAYS), "", dtype=object)
    date_index = {d: i for i, d in enumerate(dates)}
    for _, r in promo_calendar.iterrows():
        si = sku_index[r["sku_id"]]
        ti = date_index[r["date"]]
        if r["uplift_factor"] > promo_mult_mat[si, ti]:
            promo_mult_mat[si, ti] = r["uplift_factor"]
            promo_name_mat[si, ti] = r["promo_name"]
        promo_active_mat[si, ti] = True

    # ---- Decide which (sku, store) pairs are stocked, and which are phantom targets ----
    stocked_mask = rng.random((N_STORES, n_sku)) >= (1 - SKU_STOCKED_PROB)  # True = carried
    phantom_roll = rng.random((N_STORES, n_sku))
    phantom_mask = stocked_mask & (phantom_roll < PHANTOM_RATE)

    n_pairs = int(stocked_mask.sum())
    print(f"Simulating {n_pairs:,} sku-store pairs x {N_DAYS} days ...")

    # ---- Preallocate output arrays (avoids 9M+ python dict appends) ----
    max_rows = n_pairs * N_DAYS
    out_date = np.empty(max_rows, dtype="datetime64[ns]")
    out_store = np.empty(max_rows, dtype=object)
    out_sku = np.empty(max_rows, dtype=object)
    out_sold = np.empty(max_rows, dtype=np.int32)
    out_onhand = np.empty(max_rows, dtype=np.int32)
    out_promo_active = np.empty(max_rows, dtype=np.int8)
    out_promo_name = np.empty(max_rows, dtype=object)

    ground_truth_rows = []
    write_ptr = 0

    for si_store, store in enumerate(store_ids):
        sfactor = store_factor_map[store]
        row_mask = stocked_mask[si_store]
        sku_idxs_for_store = np.nonzero(row_mask)[0]
        if len(sku_idxs_for_store) == 0:
            continue

        # randomize per-sku-store demand scaling in one vectorized call
        demand_jitter = rng.uniform(0.7, 1.3, size=len(sku_idxs_for_store))
        bdemands = base_demand_arr[sku_idxs_for_store] * sfactor * demand_jitter
        keep = bdemands >= 0.05
        sku_idxs_for_store = sku_idxs_for_store[keep]
        bdemands = bdemands[keep]

        for k, si in enumerate(sku_idxs_for_store):
            sku = sku_ids[si]
            bdemand = bdemands[k]
            noise_sd = max(0.3, bdemand * 0.35)

            is_phantom = bool(phantom_mask[si_store, si])
            phantom_start = -1
            phantom_len = 0
            if is_phantom:
                phantom_len = int(rng.integers(10, 35))
                latest_start = N_DAYS - phantom_len - 1
                earliest_start = int(N_DAYS * 0.25)
                if latest_start > earliest_start:
                    phantom_start = int(rng.integers(earliest_start, latest_start))
                else:
                    is_phantom = False

            # vectorized expected demand + noise for the whole date range
            expected = bdemand * day_base_mult_arr * promo_mult_mat[si]
            noise = rng.normal(0, noise_sd, size=N_DAYS)
            true_demand = np.maximum(0.0, expected + noise)

            stock = float(rng.integers(max(5, int(bdemand * 10)), max(8, int(bdemand * 20)) + 8))
            reorder_point = max(3, bdemand * 4)
            reorder_qty = max(10, bdemand * 14)
            replenish_jitter = rng.uniform(0.8, 1.2, size=N_DAYS)

            sold_arr = np.empty(N_DAYS, dtype=np.int32)
            onhand_arr = np.empty(N_DAYS, dtype=np.int32)

            phantom_end = phantom_start + phantom_len if is_phantom else -1

            for t in range(N_DAYS):
                if is_phantom and phantom_start <= t < phantom_end:
                    sold = 0
                    shown = max(stock, 2 + (t % 8))  # system still shows stock
                else:
                    sold = int(true_demand[t]) if stock > 0 else 0
                    if sold > stock:
                        sold = int(stock)
                    stock -= sold
                    if stock <= reorder_point:
                        stock += reorder_qty * replenish_jitter[t]
                    shown = stock if stock > 0 else 0

                sold_arr[t] = sold
                onhand_arr[t] = int(round(shown))

            n = N_DAYS
            out_date[write_ptr:write_ptr + n] = dates.values
            out_store[write_ptr:write_ptr + n] = store
            out_sku[write_ptr:write_ptr + n] = sku
            out_sold[write_ptr:write_ptr + n] = sold_arr
            out_onhand[write_ptr:write_ptr + n] = onhand_arr
            out_promo_active[write_ptr:write_ptr + n] = promo_active_mat[si].astype(np.int8)
            out_promo_name[write_ptr:write_ptr + n] = promo_name_mat[si]
            write_ptr += n

            if is_phantom:
                ground_truth_rows.append({
                    "store_id": store,
                    "sku_id": sku,
                    "phantom_start_date": dates[phantom_start],
                    "phantom_end_date": dates[min(phantom_end - 1, N_DAYS - 1)],
                    "phantom_length_days": phantom_len,
                    "system_stock_during_phantom": int(onhand_arr[phantom_start]),
                    "pre_phantom_avg_daily_sales": round(float(bdemand), 2),
                })

        if (si_store + 1) % 5 == 0 or si_store == len(store_ids) - 1:
            print(f"  ... store {si_store + 1}/{len(store_ids)} done")

    sales_df = pd.DataFrame({
        "date": out_date[:write_ptr],
        "store_id": out_store[:write_ptr],
        "sku_id": out_sku[:write_ptr],
        "qty_sold": out_sold[:write_ptr],
        "qty_on_hand": out_onhand[:write_ptr],
        "promo_active": out_promo_active[:write_ptr],
        "promo_name": out_promo_name[:write_ptr],
    })
    # Stored as parquet: ~9.2M rows as CSV would be ~280MB and slow to reload on
    # every Streamlit rerun. Parquet keeps the demo snappy. load_real_data.py
    # still accepts a plain CSV from a real retailer's export.
    sales_df.to_parquet(OUT_DIR / "sales_inventory.parquet", index=False)

    gt_df = pd.DataFrame(ground_truth_rows)
    gt_df.to_csv(OUT_DIR / "ground_truth.csv", index=False)

    print(f"Stores:            {len(stores)}")
    print(f"SKUs:              {len(skus)}")
    print(f"Days:              {N_DAYS} ({dates.min().date()} -> {dates.max().date()})")
    print(f"Sales/inv rows:    {len(sales_df):,}")
    print(f"Promo events:      {len(promo_calendar):,}")
    print(f"Phantom stockouts injected (ground truth): {len(gt_df)}")
    print(f"Written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
