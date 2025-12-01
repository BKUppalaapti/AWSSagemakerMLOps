import os
import json
import boto3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import shap

from utils.logger import get_logger, cleanup_temp

# ---------------- CONFIG ----------------
BUCKET = "aws-sagemaker-end2end-project"

FEATURE_PREFIX = "data/features/"
MODEL_PREFIX = "models/"
EVAL_PREFIX = "evaluation/"

logger = get_logger("model_evaluation", s3_bucket=BUCKET)


# ------------- HELPERS: S3 ------------- #
def download_from_s3(key: str) -> str:
    """
    Download a file from S3 to temp/ and return local path.
    """
    s3 = boto3.client("s3")
    os.makedirs("temp", exist_ok=True)
    local_path = os.path.join("temp", os.path.basename(key))

    s3.download_file(BUCKET, key, local_path)
    logger.info(f"Downloaded s3://{BUCKET}/{key} -> {local_path}")
    return local_path


def upload_to_s3(local_path: str, key: str):
    """
    Upload a local file to S3.
    """
    s3 = boto3.client("s3")
    s3.upload_file(local_path, BUCKET, key)
    logger.info(f"Uploaded {local_path} -> s3://{BUCKET}/{key}")


# ---------- LOAD MODEL + DATA ---------- #
def load_model_from_s3() -> object:
    model_key = f"{MODEL_PREFIX}logreg_model.pkl"
    local_model_path = download_from_s3(model_key)
    model = joblib.load(local_model_path)
    return model


def load_test_features() -> (pd.DataFrame, pd.Series):
    test_key = f"{FEATURE_PREFIX}features_test.csv"
    local_test_path = download_from_s3(test_key)

    df = pd.read_csv(local_test_path)
    if "sentiment" not in df.columns:
        raise ValueError("Column 'sentiment' not found in test features.")

    X_test = df.drop(columns=["sentiment"])
    y_test = df["sentiment"]
    logger.info(f"Loaded test features: X_test={X_test.shape}, y_test={y_test.shape}")
    return X_test, y_test


# --------- EVALUATION METRICS ---------- #
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
    }

    logger.info(f"Eval metrics: {metrics}")

    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, cls_report, cm


def save_confusion_matrix(cm: np.ndarray, local_path: str):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["class 0", "class 1"])
    plt.yticks([0, 1], ["class 0", "class 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(local_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot -> {local_path}")


# ------------- SHAP EXPLAINABILITY ------------- #
def compute_and_save_shap(model, X_test: pd.DataFrame):
    """
    Computes SHAP values using new masker API (no warnings)
    Saves summary plot + feature importance JSON
    """
    logger.info("Computing SHAP values using new masker API...")

    # Choose a background subset (for speed)
    background = shap.sample(X_test, min(200, X_test.shape[0]))

    # Use new SHAP masker API (no warnings)
    masker = shap.maskers.Independent(background)

    explainer = shap.LinearExplainer(model, masker)

    shap_values = explainer(X_test)

    # Convert SHAP object → array
    shap_array = shap_values.values

    # ---- Save summary plot ----
    shap_plot_path = os.path.join("temp", "shap_summary.png")
    plt.figure()
    shap.summary_plot(shap_array, X_test, show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_path)
    plt.close()

    upload_to_s3(shap_plot_path, f"{EVAL_PREFIX}shap_summary.png")

    # ---- Save feature importance ----
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    feature_importance = [
        {"feature": feat, "importance": float(imp)}
        for feat, imp in sorted(zip(X_test.columns, mean_abs),
                                key=lambda x: x[1],
                                reverse=True)
    ]

    shap_json_path = os.path.join("temp", "shap_feature_importance.json")
    with open(shap_json_path, "w") as f:
        json.dump(feature_importance, f, indent=2)

    upload_to_s3(shap_json_path, f"{EVAL_PREFIX}shap_feature_importance.json")

    logger.info("SHAP explainability artifacts created and uploaded.")



# ----------------- MAIN ----------------- #
def main():
    logger.info("===== Starting Model Evaluation =====")

    try:
        # Load model and test data
        model = load_model_from_s3()
        X_test, y_test = load_test_features()

        # Evaluate metrics
        metrics, cls_report, cm = evaluate_model(model, X_test, y_test)

        os.makedirs("temp", exist_ok=True)

        # Save metrics JSON locally
        metrics_path = os.path.join("temp", "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        upload_to_s3(metrics_path, f"{EVAL_PREFIX}eval_metrics.json")

        # Save classification report
        cls_report_path = os.path.join("temp", "classification_report.json")
        with open(cls_report_path, "w") as f:
            json.dump(cls_report, f, indent=2)
        upload_to_s3(cls_report_path, f"{EVAL_PREFIX}classification_report.json")

        # Save confusion matrix image
        cm_path = os.path.join("temp", "confusion_matrix.png")
        save_confusion_matrix(cm, cm_path)
        upload_to_s3(cm_path, f"{EVAL_PREFIX}confusion_matrix.png")

        # SHAP explainability
        compute_and_save_shap(model, X_test)

        logger.info("===== Model Evaluation Completed Successfully =====")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

    # Upload log to S3 and clean up
    try:
        logger.upload_to_s3()
    except Exception as e:
        logger.error(f"Failed to upload evaluation log: {e}")

    cleanup_temp()


if __name__ == "__main__":
    main()
