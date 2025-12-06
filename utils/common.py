import os
import pandas as pd
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp():
    """Return YYYYMMDD-HHMMSS timestamp for versioning."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)
