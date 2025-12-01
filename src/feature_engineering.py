import os
import boto3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logger import get_logger, cleanup_temp


BUCKET = "aws-sagemaker-end2end-project"

PROCESSED_PREFIX = "data/processed/"
FEATURE_PREFIX = "data/features/"

logger = get_logger("feature_engineering", s3_bucket=BUCKET)


# ------------------------------------------------------
# 1. List processed CSVs in S3
# ------------------------------------------------------
def list_s3_files(bucket: str, prefix: str):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        logger.warning(f"No processed files found in s3://{bucket}/{prefix}")
        return []

    return [
        f"s3://{bucket}/{obj['Key']}"
        for obj in response["Contents"]
        if obj["Key"].endswith(".csv")
    ]


# ------------------------------------------------------
# 2. Download CSV from S3 (no s3fs required)
# ------------------------------------------------------
def load_csv(s3_url: str):
    try:
        bucket = s3_url.split("/")[2]
        key = "/".join(s3_url.split("/")[3:])

        os.makedirs("temp", exist_ok=True)
        local_path = f"temp/{os.path.basename(key)}"

        boto3.client("s3").download_file(bucket, key, local_path)
        df = pd.read_csv(local_path)

        logger.info(f"Loaded {s3_url} -> shape={df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed loading {s3_url}: {e}")
        raise


# ------------------------------------------------------
# 3. TF-IDF feature extraction
# ------------------------------------------------------
def generate_features(train_df: pd.DataFrame, test_df: pd.DataFrame):

    try:
        vectorizer = TfidfVectorizer(max_features=5000)

        X_train = vectorizer.fit_transform(train_df["content"]).toarray()
        X_test = vectorizer.transform(test_df["content"]).toarray()

        train_out = pd.DataFrame(X_train)
        train_out["sentiment"] = train_df["sentiment"].values

        test_out = pd.DataFrame(X_test)
        test_out["sentiment"] = test_df["sentiment"].values

        logger.info(
            f"Generated features -> train: {train_out.shape}, test: {test_out.shape}"
        )

        return train_out, test_out

    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        raise


# ------------------------------------------------------
# 4. Save S3 output
# ------------------------------------------------------
def save_to_s3(df: pd.DataFrame, output_key: str):
    s3 = boto3.client("s3")

    os.makedirs("temp", exist_ok=True)
    local_tmp = f"temp/{os.path.basename(output_key)}"

    df.to_csv(local_tmp, index=False)

    s3.upload_file(local_tmp, BUCKET, output_key)

    logger.info(f"Uploaded -> s3://{BUCKET}/{output_key}")

    os.remove(local_tmp)


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():

    logger.info("===== Starting Feature Engineering =====")

    files = list_s3_files(BUCKET, PROCESSED_PREFIX)
    if not files:
        logger.warning("No processed files found to create features.")
        logger.upload_to_s3()
        return

    # Expecting two processed files: train & test
    train_file = [f for f in files if "train" in f][0]
    test_file = [f for f in files if "test" in f][0]

    # Load
    train_df = load_csv(train_file)
    test_df = load_csv(test_file)

    # Feature engineering
    train_feat, test_feat = generate_features(train_df, test_df)

    # Save outputs
    train_out_key = f"{FEATURE_PREFIX}features_train.csv"
    test_out_key = f"{FEATURE_PREFIX}features_test.csv"

    save_to_s3(train_feat, train_out_key)
    save_to_s3(test_feat, test_out_key)

    logger.info("===== Feature Engineering Completed =====")

    logger.upload_to_s3()

    # Close handlers
    import logging
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    cleanup_temp()


if __name__ == "__main__":
    main()
