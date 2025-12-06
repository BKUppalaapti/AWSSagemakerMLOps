import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import os
from utils.common import ensure_dir

# Add retries to S3 client
_retry_cfg = Config(
    retries={
        "max_attempts": 10,
        "mode": "adaptive"
    }
)

s3 = boto3.client("s3", config=_retry_cfg)


def exists(bucket, key):
    """Check if an S3 key exists."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def download_from_s3(bucket, key, local_path):
    ensure_dir(local_path)

    if not exists(bucket, key):
        raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}")

    print(f"[S3 DOWNLOAD] s3://{bucket}/{key} → {local_path}")
    s3.download_file(bucket, key, local_path)
    return local_path


def upload_to_s3(local_path, bucket, key):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found for upload: {local_path}")

    print(f"[S3 UPLOAD] {local_path} → s3://{bucket}/{key}")
    ensure_dir(local_path)
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def upload_versioned(local_path, bucket, prefix):
    """Upload with timestamp-based versioning."""
    from utils.common import timestamp

    version_key = f"{prefix}{timestamp()}_{os.path.basename(local_path)}"
    return upload_to_s3(local_path, bucket, version_key)
