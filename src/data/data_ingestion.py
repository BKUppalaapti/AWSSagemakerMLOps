import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import get_logger
from utils.s3_utils import download_from_s3, upload_to_s3
from utils.common import ensure_folder
from utils.config_loader import cfg

logger = get_logger("data_ingestion")

# -----------------------------
# CONFIG
# -----------------------------
BUCKET = cfg["buckets"]["data"]

SRC_KEY = cfg["paths"]["data"]["source_csv"]
RAW_PREFIX = cfg["paths"]["data"]["raw_prefix"]

TRAIN_NAME = "tweet_emotions_train.csv"
TEST_NAME = "tweet_emotions_test.csv"


# -----------------------------
# HELPERS
# -----------------------------
def clean_text(text):
    """Basic cleaning for tweet content."""
    if not isinstance(text, str):
        return ""
    return (
        text.replace("\n", " ").replace("\r", " ").strip()
    )


def load_source_csv():
    """Download original CSV from S3 → temp/source.csv."""
    local_path = "temp/source.csv"
    ensure_folder("temp")

    logger.info(f"Downloading source CSV: s3://{BUCKET}/{SRC_KEY}")
    download_from_s3(BUCKET, SRC_KEY, local_path)

    df = pd.read_csv(local_path)
    logger.info(f"Loaded dataset: {df.shape}")
    return df


# -----------------------------
# MAIN INGEST LOGIC
# -----------------------------
def main():

    logger.info("=== Starting Data Ingestion Step ===")

    df = load_source_csv()

    # -------------------------
    # Clean text
    # -------------------------
    if "content" not in df.columns:
        raise ValueError("Input CSV must contain a 'content' column.")

    df["cleaned_content"] = df["content"].apply(clean_text)

    # -------------------------
    # Train/Test Split
    # -------------------------
    logger.info("Splitting train/test...")
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

    logger.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # -------------------------
    # Save locally
    # -------------------------
    ensure_folder("temp")

    train_local = f"temp/{TRAIN_NAME}"
    test_local = f"temp/{TEST_NAME}"

    train_df.to_csv(train_local, index=False)
    test_df.to_csv(test_local, index=False)

    # -------------------------
    # Upload to S3
    # -------------------------
    logger.info("Uploading train/test split to S3...")

    upload_to_s3(train_local, BUCKET, RAW_PREFIX + TRAIN_NAME)
    upload_to_s3(test_local, BUCKET, RAW_PREFIX + TEST_NAME)

    logger.info(f"Uploaded TRAIN → s3://{BUCKET}/{RAW_PREFIX}{TRAIN_NAME}")
    logger.info(f"Uploaded TEST  → s3://{BUCKET}/{RAW_PREFIX}{TEST_NAME}")

    logger.info("=== Data Ingestion Completed Successfully ===")


if __name__ == "__main__":
    main()
