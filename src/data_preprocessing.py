import os
import re
import boto3
import pandas as pd
from utils.logger import get_logger, cleanup_temp


BUCKET = "aws-sagemaker-end2end-project"

RAW_PREFIX = "data/raw/"           # input from ingestion step
PROCESSED_PREFIX = "data/processed/"   # output of this step

logger = get_logger("data_preprocessing", s3_bucket=BUCKET)


# ------------------------------------------------------
# 1. List raw CSV files in S3
# ------------------------------------------------------
def list_s3_files(bucket: str, prefix: str):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        logger.warning(f"No CSV files found in s3://{bucket}/{prefix}")
        return []

    return [
        f"s3://{bucket}/{obj['Key']}"
        for obj in response["Contents"]
        if obj["Key"].endswith(".csv")
    ]


# ------------------------------------------------------
# 2. Download CSV from S3 (NO s3fs needed)
# ------------------------------------------------------
def load_csv(s3_url: str):

    try:
        assert s3_url.startswith("s3://")

        bucket = s3_url.split("/")[2]
        key = "/".join(s3_url.split("/")[3:])

        os.makedirs("temp", exist_ok=True)
        local_file = f"temp/{os.path.basename(key)}"

        boto3.client("s3").download_file(bucket, key, local_file)
        df = pd.read_csv(local_file)

        logger.info(f"Loaded {s3_url} -> shape={df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load {s3_url}: {e}")
        raise


# ------------------------------------------------------
# 3. Clean text
# ------------------------------------------------------
def clean_text(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------
# 4. Full preprocessing step
# ------------------------------------------------------
def preprocess_df(df: pd.DataFrame):

    try:
        df = df.copy()

        if "content" in df.columns:
            df["content"] = df["content"].apply(clean_text)

        df["sentiment"] = df["sentiment"].astype(int)

        logger.info(f"Processed dataframe -> shape={df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error preprocessing dataframe: {e}")
        raise


# ------------------------------------------------------
# 5. Save processed CSV back to S3
# ------------------------------------------------------
def save_to_s3(df: pd.DataFrame, output_key: str):

    s3 = boto3.client("s3")

    os.makedirs("temp", exist_ok=True)
    local_tmp = f"temp/{os.path.basename(output_key)}"

    df.to_csv(local_tmp, index=False)

    s3.upload_file(local_tmp, BUCKET, output_key)

    logger.info(f"Uploaded to s3://{BUCKET}/{output_key}")

    os.remove(local_tmp)


# ------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------
def main():

    logger.info("===== Starting Data Preprocessing =====")

    # Get all train/test files created by data_ingestion
    all_files = list_s3_files(BUCKET, RAW_PREFIX)

    if not all_files:
        logger.warning("No raw files to preprocess.")
        logger.upload_to_s3()
        return

    for s3_path in all_files:
        try:
            df = load_csv(s3_path)

            processed_df = preprocess_df(df)

            file_name = os.path.basename(s3_path)
            output_key = f"{PROCESSED_PREFIX}{file_name}"

            save_to_s3(processed_df, output_key)

        except Exception as e:
            logger.error(f"Failed processing {s3_path}: {e}")

    logger.info("===== Data Preprocessing Completed =====")

    logger.upload_to_s3()

    # ---- CLOSE HANDLERS CLEANLY ----
    import logging
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # ---- CLEAN TEMP FOLDER ----
    cleanup_temp()


if __name__ == "__main__":
    main()
