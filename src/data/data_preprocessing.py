import pandas as pd
from utils.logger import get_logger
from utils.s3_utils import download_from_s3, upload_to_s3
from utils.common import ensure_folder
from utils.config_loader import cfg

logger = get_logger("data_preprocessing")

# -----------------------------
# CONFIG
# -----------------------------
BUCKET = cfg["buckets"]["data"]

RAW_PREFIX = cfg["paths"]["data"]["raw_prefix"]
PROCESSED_PREFIX = cfg["paths"]["data"]["processed_prefix"]

TRAIN_NAME = "tweet_emotions_train.csv"
TEST_NAME = "tweet_emotions_test.csv"


# -----------------------------
# HELPERS
# -----------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning + safety checks."""
    df = df.copy()

    if "cleaned_content" not in df.columns:
        raise ValueError(
            "cleaned_content column missing — ingestion step must run first."
        )

    # Force string type
    df["cleaned_content"] = df["cleaned_content"].astype(str).str.strip()

    # Remove empty rows
    df = df[df["cleaned_content"].str.len() > 0]

    # Normalize sentiment label
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    return df


# -----------------------------
# MAIN PROCESSING LOGIC
# -----------------------------
def main():

    logger.info("=== Starting Data Preprocessing Step ===")

    ensure_folder("temp")

    # -------------------------
    # Download raw splits
    # -------------------------
    logger.info(f"Downloading raw CSVs from s3://{BUCKET}/{RAW_PREFIX}")

    train_local = "temp/train_raw.csv"
    test_local = "temp/test_raw.csv"

    download_from_s3(BUCKET, RAW_PREFIX + TRAIN_NAME, train_local)
    download_from_s3(BUCKET, RAW_PREFIX + TEST_NAME, test_local)

    df_train = pd.read_csv(train_local)
    df_test = pd.read_csv(test_local)

    logger.info(f"Raw Train Shape: {df_train.shape}")
    logger.info(f"Raw Test Shape:  {df_test.shape}")

    # -------------------------
    # Preprocess data
    # -------------------------
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    logger.info(f"Processed Train Shape: {df_train.shape}")
    logger.info(f"Processed Test Shape:  {df_test.shape}")

    # -------------------------
    # Save locally
    # -------------------------
    train_processed = "temp/train_processed.csv"
    test_processed = "temp/test_processed.csv"

    df_train.to_csv(train_processed, index=False)
    df_test.to_csv(test_processed, index=False)

    # -------------------------
    # Upload to S3
    # -------------------------
    logger.info("Uploading processed files to S3...")

    upload_to_s3(train_processed, BUCKET, PROCESSED_PREFIX + TRAIN_NAME)
    upload_to_s3(test_processed, BUCKET, PROCESSED_PREFIX + TEST_NAME)

    logger.info(f"Uploaded processed train → s3://{BUCKET}/{PROCESSED_PREFIX}{TRAIN_NAME}")
    logger.info(f"Uploaded processed test  → s3://{BUCKET}/{PROCESSED_PREFIX}{TEST_NAME}")

    logger.info("=== Data Preprocessing Completed Successfully ===")


if __name__ == "__main__":
    main()
