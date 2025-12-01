import os
import json
import boto3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
)
from utils.logger import get_logger, cleanup_temp

BUCKET = "aws-sagemaker-end2end-project"

FEATURE_PREFIX = "data/features/"
TRAIN_METRICS_PREFIX = "metrics/"      # <-- TRAIN METRICS
EVAL_METRICS_PREFIX = "evaluation/"    # <-- EVAL METRICS (used by model_evaluation.py)
MODEL_PREFIX = "models/"

logger = get_logger("model_train", s3_bucket=BUCKET)


# --------- DOWNLOAD FILES FROM S3 ----------
def download_from_s3(key: str):
    s3 = boto3.client("s3")
    os.makedirs("temp", exist_ok=True)
    local_path = f"temp/{os.path.basename(key)}"
    s3.download_file(BUCKET, key, local_path)
    return local_path


def load_features():
    train_key = f"{FEATURE_PREFIX}features_train.csv"
    test_key = f"{FEATURE_PREFIX}features_test.csv"

    logger.info("Downloading feature files from S3...")

    train_local = download_from_s3(train_key)
    test_local = download_from_s3(test_key)

    train_df = pd.read_csv(train_local)
    test_df = pd.read_csv(test_local)

    logger.info(f"Loaded train features: {train_df.shape}")
    logger.info(f"Loaded test features: {test_df.shape}")

    return train_df, test_df


# --------- TRAIN MODEL ----------
def train_model(train_df, test_df):
    X_train = train_df.drop(columns=["sentiment"])
    y_train = train_df["sentiment"]

    X_test = test_df.drop(columns=["sentiment"])
    y_test = test_df["sentiment"]

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # TRAIN metrics
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "auc": roc_auc_score(y_train, y_train_prob)
    }

    # EVAL metrics (saved separately by eval script, but used internally here)
    eval_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "auc": roc_auc_score(y_test, y_test_prob)
    }

    logger.info("Model trained successfully.")
    return model, train_metrics, eval_metrics


# --------- SAVE TRAIN METRICS TO S3 ----------
def save_train_metrics_to_s3(train_metrics):
    s3 = boto3.client("s3")

    os.makedirs("temp", exist_ok=True)
    train_path = "temp/train_metrics.json"

    # Save locally
    with open(train_path, "w") as f:
        json.dump(train_metrics, f)

    # Upload only TRAIN metrics to metrics/ folder
    s3.upload_file(train_path, BUCKET, f"{TRAIN_METRICS_PREFIX}train_metrics.json")

    logger.info("Uploaded train metrics → S3")
    os.remove(train_path)


# --------- SAVE MODEL PKL TO S3 ----------
def save_model_to_s3(model):
    import joblib
    s3 = boto3.client("s3")

    os.makedirs("temp", exist_ok=True)
    model_path = "temp/logreg_model.pkl"

    joblib.dump(model, model_path)

    s3.upload_file(model_path, BUCKET, f"{MODEL_PREFIX}logreg_model.pkl")

    logger.info("Uploaded model → S3")
    os.remove(model_path)


# --------- MAIN ----------
def main():
    logger.info("======== Starting MODEL TRAIN ========")

    train_df, test_df = load_features()
    model, train_metrics, eval_metrics = train_model(train_df, test_df)

    save_train_metrics_to_s3(train_metrics)
    save_model_to_s3(model)

    logger.info("======== Model Training Completed ========")

    logger.upload_to_s3()
    cleanup_temp()


if __name__ == "__main__":
    main()
