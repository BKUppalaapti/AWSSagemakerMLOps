import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

from utils.logger import get_logger
from utils.s3_utils import download_from_s3, upload_to_s3
from utils.common import ensure_folder
from utils.config_loader import cfg

logger = get_logger("feature_engineering")

# -----------------------------
# CONFIG
# -----------------------------
BUCKET = cfg["buckets"]["data"]

PROCESSED_PREFIX = cfg["paths"]["data"]["processed_prefix"]
FEATURES_PREFIX = cfg["paths"]["data"]["features_prefix"]
VECTORIZER_PREFIX = cfg["paths"]["artifacts"]["vectorizer_prefix"]

TRAIN_NAME = "tweet_emotions_train.csv"
TEST_NAME = "tweet_emotions_test.csv"

TRAIN_FEATURE_NAME = "features_train.npz"
TEST_FEATURE_NAME = "features_test.npz"
VECTORIZER_NAME = "tfidf_vectorizer.pkl"


# -----------------------------
# LOAD PROCESSED CSVs
# -----------------------------
def load_processed_data():
    ensure_folder("temp")

    train_local = "temp/processed_train.csv"
    test_local = "temp/processed_test.csv"

    logger.info("Downloading processed train/test from S3...")

    download_from_s3(BUCKET, PROCESSED_PREFIX + TRAIN_NAME, train_local)
    download_from_s3(BUCKET, PROCESSED_PREFIX + TEST_NAME, test_local)

    df_train = pd.read_csv(train_local)
    df_test = pd.read_csv(test_local)

    logger.info(f"Train shape: {df_train.shape}")
    logger.info(f"Test shape:  {df_test.shape}")

    if "cleaned_content" not in df_train.columns:
        raise ValueError("Column 'cleaned_content' missing — preprocessing step failed.")

    return df_train, df_test


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def create_tfidf(train_df, test_df):
    logger.info("Fitting TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_df["cleaned_content"])
    X_test = vectorizer.transform(test_df["cleaned_content"])

    logger.info(f"TF-IDF Train matrix: {X_train.shape}")
    logger.info(f"TF-IDF Test matrix:  {X_test.shape}")

    return X_train, X_test, vectorizer


# -----------------------------
# SAVE + UPLOAD FEATURES
# -----------------------------
def save_and_upload_features(X_train, X_test, vectorizer):
    ensure_folder("temp")

    train_path = "temp/" + TRAIN_FEATURE_NAME
    test_path = "temp/" + TEST_FEATURE_NAME
    vec_path = "temp/" + VECTORIZER_NAME

    # Save sparse matrices
    sparse.save_npz(train_path, X_train)
    sparse.save_npz(test_path, X_test)

    # Save vectorizer
    joblib.dump(vectorizer, vec_path)

    logger.info("Uploading features + vectorizer to S3...")

    upload_to_s3(train_path, BUCKET, FEATURES_PREFIX + TRAIN_FEATURE_NAME)
    upload_to_s3(test_path, BUCKET, FEATURES_PREFIX + TEST_FEATURE_NAME)
    upload_to_s3(vec_path, BUCKET, VECTORIZER_PREFIX + VECTORIZER_NAME)

    logger.info("Feature files uploaded successfully.")


# -----------------------------
# MAIN
# -----------------------------
def main():
    try:
        logger.info("=== Starting Feature Engineering Step ===")

        train_df, test_df = load_processed_data()

        X_train, X_test, vectorizer = create_tfidf(train_df, test_df)

        save_and_upload_features(X_train, X_test, vectorizer)

        logger.info("=== Feature Engineering Completed Successfully ===")

    except Exception as e:
        logger.error(f"FEATURE ENGINEERING FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
