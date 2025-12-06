import os
import joblib

from utils.common import ensure_folder
from utils.s3_utils import download_from_s3
from utils.config_loader import cfg
from utils.logger import get_logger


logger = get_logger("model_predictor")

# -------------------------------------------------------
# Config
# -------------------------------------------------------
BUCKET = cfg["buckets"]["data"]
ARTIFACT_MODELS = cfg["paths"]["artifacts"]["models_prefix"]
ARTIFACT_VECTOR = cfg["paths"]["artifacts"]["vectorizer_prefix"]

MODEL_FILE = cfg["model"]["name"]               # logreg_model.pkl
VECTORIZER_FILE = "tfidf_vectorizer.pkl"        # vectorizer


class ModelPredictor:
    """
    Loads model + vectorizer (from local OR S3)
    and performs sentiment prediction on text input.
    """

    def __init__(self):
        logger.info("🔹 Initializing ModelPredictor...")

        ensure_folder("temp")

        # ------------------------------
        # Load Model
        # ------------------------------
        model_path = f"temp/{MODEL_FILE}"

        if not os.path.exists(model_path):
            logger.info("Downloading Model from S3...")
            download_from_s3(
                BUCKET,
                ARTIFACT_MODELS + MODEL_FILE,
                model_path
            )

        self.model = joblib.load(model_path)
        logger.info("✔ Loaded model successfully.")

        # ------------------------------
        # Load Vectorizer
        # ------------------------------
        vec_path = f"temp/{VECTORIZER_FILE}"

        if not os.path.exists(vec_path):
            logger.info("Downloading Vectorizer from S3...")
            download_from_s3(
                BUCKET,
                ARTIFACT_VECTOR + VECTORIZER_FILE,
                vec_path
            )

        self.vectorizer = joblib.load(vec_path)
        logger.info("✔ Loaded vectorizer successfully.")

    # ---------------------------------------------------
    # Single Prediction
    # ---------------------------------------------------
    def predict(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return {"error": "Invalid input text."}

        cleaned = text.strip()

        X = self.vectorizer.transform([cleaned])
        pred = self.model.predict(X)[0]

        return {
            "input": cleaned,
            "prediction": str(pred)
        }

    # ---------------------------------------------------
    # Batch Prediction
    # ---------------------------------------------------
    def predict_batch(self, texts):
        if not isinstance(texts, list):
            return {"error": "Input must be a list of strings."}

        cleaned = [str(t).strip() for t in texts]

        X = self.vectorizer.transform(cleaned)
        preds = self.model.predict(X)

        return {
            "inputs": cleaned,
            "predictions": preds.tolist()
        }


if __name__ == "__main__":
    mp = ModelPredictor()

    print(mp.predict("I love AWS MLOps!"))
    print(mp.predict_batch(["This is fantastic!", "Terrible experience"]))
