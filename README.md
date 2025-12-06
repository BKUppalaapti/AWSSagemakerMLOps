# Project Objective:

Build a fully working, production-style MLOps pipeline on AWS

Using:

SageMaker Processing Jobs (ingest → preprocess → feature engineering)

SageMaker Training Job (train model with your Docker container)

SageMaker Processing Job (evaluate the model)

Custom Docker container containing your entire project code

S3 storage for data, features, models, metrics

CloudWatch logging for all steps

Clean project structure so you can maintain and scale easily

✔ Ensure the pipeline runs end-to-end automatically or step-by-step, with no manual fixes

Meaning:

Each job should run with zero hardcoded paths

Each step reads from S3 and writes back to the correct S3 folder

The Docker container contains the right file structure for training/evaluation

Logging is readable from CloudWatch

Errors are minimal and easy to debug


# Project Struxcture

AWSSagemakerMLOps/
│
├── config/
│   ├── __init__.py
│   ├── params.yaml
│
├── logs/
│   ├── data_ingestion/
│   ├── data_preprocessing/
│   ├── feature_engineering/
│   ├── model_train/
│   └── model_evaluation/
│
├── sm_jobs/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── eval_job.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── full_pipeline.py
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── ingest_job.py
│   │   ├── preprocess_job.py
│   │   └── feature_eng_job.py
│   │
│   ├── training/
│       ├── __init__.py
│       └── train_job.py
│
├── scripts/
│   ├── create_structure.py
│   └── run_local_pipeline.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── model/
│       ├── __init__.py
│       ├── model_train.py
│       ├── model_evaluation.py
│       └── model_predictor.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── s3_utils.py
│   ├── common.py
│   └── config_loader.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── entrypoint.py   (unused now)


# S3 folder Structure:

aws-sagemaker-end2end-project/
│
├── data/
│   ├── sourcedata/
│   │     └── tweet_emotions.csv
│   │
│   ├── raw/
│   │   ├── tweet_emotions_train.csv
│   │   └── tweet_emotions_test.csv
│   │
│   ├── processed/
│   │   ├── tweet_emotions_train.csv
│   │   └── tweet_emotions_test.csv
│   │
│   └── features/
│       ├── features_train.npz
│       └── features_test.npz
│
├── models/
│   ├── v1/
│   │   └── logreg_model.pkl
│   ├── v2/
│   │   └── logreg_model.pkl
│   └── logreg_model.pkl      ← latest model stored directly here
│
├── metrics/
│   ├── v1/
│   │   └── train_metrics.json
│   ├── v2/
│       └── train_metrics.json
│   └── train_metrics.json    ← latest metrics
│
├── evaluation/
│   ├── v1/
│   │   ├── eval_metrics.json
│   │   ├── classification_report.json
│   │   └── confusion_matrix.png
│   └── eval_metrics.json     ← latest evaluation
│
└── logs/
    ├── data_ingestion/
    ├── data_preprocessing/
    ├── feature_engineering/
    ├── model_train/
    └── model_evaluation/

Data flow steps per scripts:
1️⃣ DATA INGESTION
----------------------------------------
Input:
    S3 → data/sourcedata/tweet_emotions.csv

Process:
    - Clean text
    - Split into train/test (80/20)

Output:
    S3 → data/raw/
        ├── tweet_emotions_train.csv
        └── tweet_emotions_test.csv


2️⃣ DATA PREPROCESSING
----------------------------------------
Input:
    S3 → data/raw/
        ├── tweet_emotions_train.csv
        └── tweet_emotions_test.csv

Process:
    - Basic cleaning
    - Standardize columns

Output:
    S3 → data/processed/
        ├── tweet_emotions_train.csv
        └── tweet_emotions_test.csv


3️⃣ FEATURE ENGINEERING
----------------------------------------
Input:
    S3 → data/processed/
        ├── tweet_emotions_train.csv
        └── tweet_emotions_test.csv

Process:
    - TF-IDF vectorizer fit (train)
    - Transform train & test
    - Save sparse matrices (.npz)
    - Save vectorizer.pkl

Output:
    S3 → data/features/
        ├── features_train.npz
        └── features_test.npz

    S3 → artifacts/vectorizer/
        └── tfidf_vectorizer.pkl


4️⃣ MODEL TRAINING
----------------------------------------
Input:
    S3 →
        data/features/
            ├── features_train.npz
            └── features_test.npz
        data/processed/
            ├── tweet_emotions_train.csv  (for labels)
            └── tweet_emotions_test.csv

Process:
    - Load TF-IDF vectors
    - Train LogisticRegression
    - Predict on test
    - Produce metrics

Output:
    S3 → models/
        └── logreg_model.pkl

    S3 → metrics/
        └── train_metrics.json


5️⃣ MODEL EVALUATION
----------------------------------------
Input:
    S3 →
        models/logreg_model.pkl
        data/features/features_test.npz
        data/processed/tweet_emotions_test.csv

Process:
    - Evaluate model using held-out test data
    - Compute accuracy, F1, classification report
    - Generate confusion matrix plot

Output:
    S3 → evaluation/
        ├── eval_metrics.json
        ├── classification_report.json
        └── confusion_matrix.png


sagemaker-pipeline-mlops-end2end/
│
├── pipelines/
│   ├── full-pipeline/
│   │     ├── code/
│   │     │    └── full_pipeline.py
│   │     ├── execution/
│   │     ├── cache/
│   │     └── artifacts/
│
├── processing/
│   ├── ingest/
│   │     └── step artifacts
│   ├── preprocess/
│   │     └── step artifacts
│   └── feature_eng/
│         └── step artifacts
│
├── training/
│   ├── model/
│   │     └── step artifacts (temporary)
│   └── metrics/
│         └── training metrics from estimator
│
├── evaluation/
│   └── step artifacts (temporary)  
│
├── models/
│   ├── model.tar.gz   (SageMaker model registry)
│   └── approved/
│         └── production models
│
└── pipeline_logs/
      └── CloudWatch-linked logs for pipeline executions


AWS IAM roles:
SageMakerExecutionRole
SageMakerPipelineRole

ECR Repo ARN: 975248345240.dkr.ecr.us-east-1.amazonaws.com/mlops/mlmodels:latest