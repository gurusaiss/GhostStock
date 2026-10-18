"""
load_real_data.py — Loader + cleaner for a real retailer's CSV export.

Accepts a CSV with flexible / messy column naming (common real-world variants)
and normalizes it to the schema the detector expects:

    date, store_id, sku_id, qty_sold, qty_on_hand[, promo_active, promo_name]

Usage:
    from load_real_data import load_real_csv
    df = load_real_csv("my_export.csv")

Or standalone:
    python load_real_data.py path/to/export.csv
"""

import re
import pandas as pd
import numpy as np

# Candidate column names -> canonical name. Matching is case/space/punct-insensitive.
COLUMN_ALIASES = {
    "sku_id": ["sku", "sku_id", "sku_code", "item_code", "item_id", "itemcode",
               "product_code", "product_id", "article_code", "article"],
    "store_id": ["store", "store_id", "store_code", "shop", "shop_id", "outlet",
                 "outlet_id", "branch", "branch_id"],
    "date": ["date", "txn_date", "transaction_date", "sale_date", "bill_date", "day"],
    "qty_sold": ["qty_sold", "quantity_sold", "units_sold", "sales_qty", "sold_qty",
                 "qty", "quantity", "units"],
    "qty_on_hand": ["qty_on_hand", "stock_on_hand", "on_hand", "soh", "system_stock",
                     "stock", "inventory", "closing_stock", "current_stock"],
    "promo_active": ["promo_active", "is_promo", "promo", "on_promo"],
    "promo_name": ["promo_name", "promotion", "scheme", "offer"],
}


def _normalize(colname: str) -> str:
    return re.sub(r"[^a-z0-9]", "", colname.lower())


def _map_columns(df: pd.DataFrame) -> dict:
    normalized_lookup = {_normalize(c): c for c in df.columns}
    mapping = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            na = _normalize(alias)
            if na in normalized_lookup:
                mapping[canonical] = normalized_lookup[na]
                break
    return mapping


def load_real_csv(path_or_buffer) -> pd.DataFrame:
    """Load a real retailer's CSV export, cleaning it into the canonical schema.

    Returns a DataFrame with columns:
        date (datetime64), store_id (str), sku_id (str),
        qty_sold (int, >=0), qty_on_hand (int, >=0),
        promo_active (0/1), promo_name (str)
    """
    raw = pd.read_csv(path_or_buffer)
    raw.columns = [c.strip() for c in raw.columns]

    mapping = _map_columns(raw)
    required = ["sku_id", "store_id", "date", "qty_sold", "qty_on_hand"]
    missing = [r for r in required if r not in mapping]
    if missing:
        raise ValueError(
            f"Could not find columns for: {missing}. "
            f"Available columns were: {list(raw.columns)}. "
            f"Rename them to something like sku_code/item_code, store, date, "
            f"qty_sold, qty_on_hand, or add an alias to COLUMN_ALIASES."
        )

    df = pd.DataFrame()
    # dayfirst=True by default (DD-MM-YYYY is the common convention for Indian
    # retailer exports); retry the other way if that parse mostly fails.
    df["date"] = pd.to_datetime(raw[mapping["date"]], errors="coerce", dayfirst=True)
    if df["date"].isna().mean() > 0.3:
        df["date"] = pd.to_datetime(raw[mapping["date"]], errors="coerce", dayfirst=False)

    df["store_id"] = raw[mapping["store_id"]].astype(str).str.strip().str.upper()
    df["sku_id"] = raw[mapping["sku_id"]].astype(str).str.strip().str.upper()

    df["qty_sold"] = pd.to_numeric(raw[mapping["qty_sold"]], errors="coerce")
    df["qty_on_hand"] = pd.to_numeric(raw[mapping["qty_on_hand"]], errors="coerce")

    if "promo_active" in mapping:
        df["promo_active"] = (
            pd.to_numeric(raw[mapping["promo_active"]], errors="coerce").fillna(0) > 0
        ).astype(int)
    else:
        df["promo_active"] = 0

    if "promo_name" in mapping:
        df["promo_name"] = raw[mapping["promo_name"]].fillna("").astype(str)
    else:
        df["promo_name"] = ""

    # ---- basic cleaning ----
    n_before = len(df)
    df = df.dropna(subset=["date", "store_id", "sku_id"])
    # missing sales/stock -> assume 0 sold, keep last-known or 0 on-hand
    df["qty_sold"] = df["qty_sold"].fillna(0).clip(lower=0)
    df["qty_on_hand"] = df["qty_on_hand"].fillna(0).clip(lower=0)

    # drop obvious junk rows (both zero and clearly placeholder codes like "NA", "-")
    df = df[~df["store_id"].isin(["NAN", "NA", "-", "NONE", ""])]
    df = df[~df["sku_id"].isin(["NAN", "NA", "-", "NONE", ""])]

    # dedupe exact duplicate rows (common in POS exports with repeated batch uploads)
    df = df.drop_duplicates(subset=["date", "store_id", "sku_id"], keep="last")

    df["qty_sold"] = df["qty_sold"].round().astype(int)
    df["qty_on_hand"] = df["qty_on_hand"].round().astype(int)
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)

    n_after = len(df)
    dropped = n_before - n_after
    if dropped:
        print(f"[load_real_data] Cleaned input: dropped {dropped} unusable rows "
              f"out of {n_before} ({dropped / n_before:.1%}).")

    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_real_data.py path/to/export.csv")
        sys.exit(1)
    out = load_real_csv(sys.argv[1])
    print(out.head(20))
    print(f"\nLoaded {len(out):,} rows, {out['store_id'].nunique()} stores, "
          f"{out['sku_id'].nunique()} skus, {out['date'].min().date()} -> {out['date'].max().date()}")
