"""
data_io.py — shared helpers for loading the synthetic (or uploaded real) dataset.
Used by detector.py, evaluate.py and app.py so they all agree on the schema.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_synthetic():
    """Load the generated synthetic dataset. Returns (sales_df, stores_df, sku_df)."""
    sales_path_parquet = DATA_DIR / "sales_inventory.parquet"
    sales_path_csv = DATA_DIR / "sales_inventory.csv"
    if sales_path_parquet.exists():
        sales = pd.read_parquet(sales_path_parquet)
    elif sales_path_csv.exists():
        sales = pd.read_csv(sales_path_csv, parse_dates=["date"])
    else:
        raise FileNotFoundError(
            "No synthetic dataset found. Run `python generate_data.py` first."
        )
    sales["date"] = pd.to_datetime(sales["date"])
    sales["promo_name"] = sales["promo_name"].fillna("")

    stores = pd.read_csv(DATA_DIR / "stores.csv")
    sku_master = pd.read_csv(DATA_DIR / "sku_master.csv")
    return sales, stores, sku_master


def load_ground_truth():
    """Hidden evaluation-only labels. Never passed to the detector."""
    path = DATA_DIR / "ground_truth.csv"
    gt = pd.read_csv(path, parse_dates=["phantom_start_date", "phantom_end_date"])
    return gt


def normalize_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure an externally-loaded (real retailer) dataframe has the columns the
    detector needs, filling optional ones if missing."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "promo_active" not in df.columns:
        df["promo_active"] = 0
    if "promo_name" not in df.columns:
        df["promo_name"] = ""
    df["promo_name"] = df["promo_name"].fillna("")
    return df
