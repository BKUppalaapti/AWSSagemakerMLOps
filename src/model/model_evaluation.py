import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from utils.logger import get_logger
from utils.common import ensure_folder
from utils.s3_utils import download_from_s3, upload_to_s3
from utils.config_loader import cfg


logger = get_logger("model_evaluation")

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
BUCKET = cfg["buckets"]["data"]

FEATURE_PREFIX = cfg["paths"]["data"]["features_prefix"]
PROCESSED_PREFIX = cfg["paths"]["data"]["processed_prefix"]

ARTIFACT_MODELS = cfg["paths"]["artifacts"]["models_prefix"]
ARTIFACT_EVAL = cfg["paths"]["artifacts"]["evaluation_prefix"]
ARTIFACT_VECTOR = cfg["paths"]["artifacts"]["vectorizer_prefix"]

MODEL_FILE = cfg["model"]["name"]

TEST_CSV = "tweet_emotions_test.csv"
TEST_NPZ = "features_test.npz"
VECTORIZER_NAME = "tfidf_vectorizer.pkl"


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def download_npz(key, local_name):
    local_path = f"temp/{local_name}"
    download_from_s3(BUCKET, key, local_path)
    return local_path


def load_vectorizer():
    ensure_folder("temp")
    vec_local = "temp/vectorizer.pkl"

    download_from_s3(BUCKET, ARTIFACT_VECTOR + VECTORIZER_NAME, vec_local)
    return joblib.load(vec_local)


def load_model():
    ensure_folder("temp")
    model_local = f"temp/{MODEL_FILE}"

    download_from_s3(BUCKET, ARTIFACT_MODELS + MODEL_FILE, model_local)
    return joblib.load(model_local)


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df["sentiment"].astype(str).values


# -------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------
def main():

    ensure_folder("temp")

    try:
        logger.info("🔍 Starting model evaluation...")

        # -------------------------------------------------
        # 1. Load feature matrix (.npz)
        # -------------------------------------------------
        logger.info("Downloading test features (.npz)...")

        test_npz_local = download_npz(FEATURE_PREFIX + TEST_NPZ, TEST_NPZ)
        X_test = sparse.load_npz(test_npz_local)

        logger.info(f"Loaded test feature matrix → {X_test.shape}")

        # -------------------------------------------------
        # 2. Load test labels
        # -------------------------------------------------
        logger.info("Downloading processed test CSV...")

        test_csv_local = "temp/test_labels.csv"
        download_from_s3(BUCKET, PROCESSED_PREFIX + TEST_CSV, test_csv_local)

        y_test = load_labels(test_csv_local)

        # -------------------------------------------------
        # 3. Load model & vectorizer
        # -------------------------------------------------
        logger.info("Loading model + vectorizer...")

        model = load_model()
        vectorizer = load_vectorizer()

        # -------------------------------------------------
        # 4. Predict
        # -------------------------------------------------
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        cls_report = classification_report(y_test, preds, output_dict=True)

        logger.info(f"Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

        metrics = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "classification_report": cls_report,
        }

        # -------------------------------------------------
        # 5. Confusion Matrix
        # -------------------------------------------------
        cm = confusion_matrix(y_test, preds)

        plt.figure(figsize=(7, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()

        cm_local = "temp/confusion_matrix.png"
        plt.savefig(cm_local)

        # -------------------------------------------------
        # 6. Upload evaluation results to S3
        # -------------------------------------------------
        metrics_local = "temp/eval_metrics.json"
        with open(metrics_local, "w") as f:
            json.dump(metrics, f, indent=4)

        report_local = "temp/classification_report.json"
        with open(report_local, "w") as f:
            json.dump(cls_report, f, indent=4)

        upload_to_s3(metrics_local, BUCKET, ARTIFACT_EVAL + "eval_metrics.json")
        upload_to_s3(report_local, BUCKET, ARTIFACT_EVAL + "classification_report.json")
        upload_to_s3(cm_local, BUCKET, ARTIFACT_EVAL + "confusion_matrix.png")

        logger.info("🎉 Model evaluation completed successfully!")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
