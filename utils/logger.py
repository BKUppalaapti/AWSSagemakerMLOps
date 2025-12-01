import logging
import os
import boto3
from datetime import datetime
import shutil


def get_logger(script_name: str, s3_bucket: str):

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Ensure logs/<script_name>/ folder exists
    log_dir = f"logs/{script_name}"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{script_name}_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def upload_to_s3():
        try:
            s3 = boto3.client("s3")
            s3_key = f"logs/{script_name}/{os.path.basename(log_file)}"
            s3.upload_file(log_file, s3_bucket, s3_key)
            logger.info(f"Uploaded log to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload log: {e}")

    logger.upload_to_s3 = upload_to_s3
    logger.log_file = log_file

    return logger


def cleanup_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
