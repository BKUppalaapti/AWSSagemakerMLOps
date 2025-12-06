import yaml
import os

CONFIG_PATH = os.path.join(os.getcwd(), "config", "params.yaml")

def _normalize_prefix(p):
    """Ensure S3 prefixes always end with '/'."""
    if p is None:
        return ""
    return p if p.endswith("/") else p + "/"

def _validate_keys(cfg):
    required = [
        "project", "aws", "buckets", "paths",
        "model", "features", "sagemaker"
    ]
    for key in required:
        if key not in cfg:
            raise KeyError(f"[CONFIG ERROR] Missing required key: {key}")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    _validate_keys(cfg)

    # Normalize S3 prefixes
    for section in ["data", "artifacts"]:
        if section in cfg["paths"]:
            for key, value in cfg["paths"][section].items():
                if isinstance(value, str):
                    cfg["paths"][section][key] = _normalize_prefix(value)

    return cfg


cfg = load_config()

# Print minimal summary (useful for debugging)
print(f"[CONFIG LOADED] Project: {cfg['project']['name']} | Region: {cfg['project']['region']}")
