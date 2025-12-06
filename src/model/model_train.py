import os
import json
import joblib
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.logger import get_logger
from utils.common import ensure_folder
from utils.s3_utils import download_from_s3, upload_to_s3
from utils.config_loader import cfg

logger = get_logger("model_train")

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
BUCKET = cfg["buckets"]["data"]

FEATURE_PREFIX = cfg["paths"]["data"]["features_prefix"]
PROCESSED_PREFIX = cfg["paths"]["data"]["processed_prefix"]

ARTIFACT_MODELS = cfg["paths"]["artifacts"]["models_prefix"]
ARTIFACT_METRICS = cfg["paths"]["artifacts"]["metrics_prefix"]
ARTIFACT_VECTOR = cfg["paths"]["artifacts"]["vectorizer_prefix"]

MODEL_FILE = cfg["model"]["name"]

TRAIN_CSV = "tweet_emotions_train.csv"
TEST_CSV = "tweet_emotions_test.csv"

TRAIN_NPZ = "features_train.npz"
TEST_NPZ = "features_test.npz"
VECTORIZER_NAME = "tfidf_vectorizer.pkl"


# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def download_npz(key, local_name):
    local_path = f"temp/{local_name}"
    download_from_s3(BUCKET, key, local_path)
    return local_path


def load_labels(local_csv_path):
    df = pd.read_csv(local_csv_path)
    return df["sentiment"].astype(str).values


def load_vectorizer():
    vec_local = "temp/vectorizer.pkl"
    download_from_s3(BUCKET, ARTIFACT_VECTOR + VECTORIZER_NAME, vec_local)
    return joblib.load(vec_local)


# ------------------------------------------------------
# MAIN TRAINING LOGIC
# ------------------------------------------------------
def main():

    ensure_folder("temp")

    try:
        logger.info("🚀 Starting Model Training Step...")

        # ----------------------------------------
        # 1. Download feature matrices (.npz)
        # ----------------------------------------
        logger.info("Downloading TF-IDF feature matrices...")

        train_npz = download_npz(FEATURE_PREFIX + TRAIN_NPZ, TRAIN_NPZ)
        test_npz = download_npz(FEATURE_PREFIX + TEST_NPZ, TEST_NPZ)

        X_train = sparse.load_npz(train_npz)
        X_test = sparse.load_npz(test_npz)

        logger.info(f"Loaded features → Train: {X_train.shape}, Test: {X_test.shape}")

        # ----------------------------------------
        # 2. Download processed CSV for labels
        # ----------------------------------------
        logger.info("Downloading processed train/test CSVs for labels...")

        train_csv_local = "temp/train_labels.csv"
        test_csv_local = "temp/test_labels.csv"

        download_from_s3(BUCKET, PROCESSED_PREFIX + TRAIN_CSV, train_csv_local)
        download_from_s3(BUCKET, PROCESSED_PREFIX + TEST_CSV, test_csv_local)

        y_train = load_labels(train_csv_local)
        y_test = load_labels(test_csv_local)

        # ----------------------------------------
        # 3. Load vectorizer (for predictor)
        # ----------------------------------------
        vectorizer = load_vectorizer()
        logger.info("Vectorizer loaded successfully.")

        # ----------------------------------------
        # 4. Train Logistic Regression
        # ----------------------------------------
        logger.info("Training LogisticRegression model...")

        model = LogisticRegression(max_iter=2000, n_jobs=-1)
        model.fit(X_train, y_train)

        # ----------------------------------------
        # 5. Evaluate Model
        # ----------------------------------------
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        cls_report = classification_report(y_test, preds, output_dict=True)

        logger.info(f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": cls_report,
        }

        # ----------------------------------------
        # 6. Save model locally + upload to S3
        # ----------------------------------------
        ensure_folder("temp")

        model_local = f"temp/{MODEL_FILE}"
        joblib.dump(model, model_local)

        upload_to_s3(model_local, BUCKET, ARTIFACT_MODELS + MODEL_FILE)
        logger.info("Uploaded model to S3.")

        # ----------------------------------------
        # 7. Save metrics JSON
        # ----------------------------------------
        metrics_local = "temp/train_metrics.json"
        with open(metrics_local, "w") as f:
            json.dump(metrics, f, indent=4)

        upload_to_s3(metrics_local, BUCKET, ARTIFACT_METRICS + "train_metrics.json")
        logger.info("Uploaded training metrics to S3.")

        logger.info("🎉 Model Training Completed Successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
