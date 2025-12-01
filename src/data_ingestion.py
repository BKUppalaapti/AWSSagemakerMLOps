import os
import html
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger, cleanup_temp


BUCKET = "aws-sagemaker-end2end-project"
SOURCE_PREFIX = "data/sourcedata/"
RAW_PREFIX = "data/raw/"

logger = get_logger("data_ingestion", s3_bucket=BUCKET)


# 1. List CSVs in S3
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


# 2. Load CSV from S3
def load_csv(s3_url: str):
    try:
        assert s3_url.startswith("s3://")
        bucket = s3_url.split("/")[2]
        key = "/".join(s3_url.split("/")[3:])

        os.makedirs("temp", exist_ok=True)
        local_path = f"temp/{os.path.basename(key)}"

        boto3.client("s3").download_file(bucket, key, local_path)

        df = pd.read_csv(local_path)
        logger.info(f"Loaded {s3_url}, shape={df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load {s3_url}: {e}")
        raise


# 3. Preprocess
def preprocess(df: pd.DataFrame):
    df = df.copy()

    if "tweet_id" in df.columns:
        df = df.drop(columns=["tweet_id"])

    df["content"] = df["content"].apply(lambda x: html.unescape(str(x)))

    df["sentiment"] = df["sentiment"].astype(str)
    df = df[df["sentiment"].isin(["happiness", "sadness"])]

    df["sentiment"] = LabelEncoder().fit_transform(df["sentiment"])

    logger.info(f"Preprocessed data: shape={df.shape}")
    return df


# 4. Save train/test to S3
def save_to_s3(train_df, test_df, base_name: str):
    s3 = boto3.client("s3")

    train_key = f"{RAW_PREFIX}{base_name}_train.csv"
    test_key = f"{RAW_PREFIX}{base_name}_test.csv"

    os.makedirs("temp", exist_ok=True)
    train_tmp = f"temp/{base_name}_train.csv"
    test_tmp = f"temp/{base_name}_test.csv"

    train_df.to_csv(train_tmp, index=False)
    test_df.to_csv(test_tmp, index=False)

    s3.upload_file(train_tmp, BUCKET, train_key)
    s3.upload_file(test_tmp, BUCKET, test_key)

    logger.info(f"Uploaded:\n - s3://{BUCKET}/{train_key}\n - s3://{BUCKET}/{test_key}")

    os.remove(train_tmp)
    os.remove(test_tmp)


# MAIN
def main():
    logger.info("===== Starting Data Ingestion =====")

    files = list_s3_files(BUCKET, SOURCE_PREFIX)
    if not files:
        logger.warning("No source CSV files found.")
        return

    for file_path in files:
        try:
            base_name = os.path.basename(file_path).replace(".csv", "")

            df = load_csv(file_path)
            df = preprocess(df)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            save_to_s3(train_df, test_df, base_name)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info("===== Data Ingestion Completed =====")

    # Upload log
    logger.upload_to_s3()

    # Close file handlers
    import logging
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Clean temp folder
    cleanup_temp()


if __name__ == "__main__":
    main()
