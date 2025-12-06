import argparse

# Correct imports matching your Docker structure
from data.data_ingestion import run_ingestion
from data.data_preprocessing import run_preprocessing
from data.feature_engineering import run_feature_engineering

from model.model_train import train
from model.model_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingestion()

    elif args.mode == "preprocess":
        run_preprocessing()

    elif args.mode == "feature_eng":
        run_feature_engineering()

    elif args.mode == "train":
        train()

    elif args.mode == "eval":
        evaluate()

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
