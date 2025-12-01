# AWS MLOps End to End Project on AWS services
    - Production-Ready Machine Learning Pipeline using AWS Native Service
    - This project demonstrates a complete, production-grade MLOps system built entirely on AWS-native services, without DVC, Dagshub, MLFlow, or external platforms.

🧩 Architecture Overview

- SageMaker Pipelines
- SageMaker Processing Jobs
- SageMaker Training Jobs
- SageMaker Model Registry
- SageMaker Real-Time Endpoints
- S3 for all data + model storage
- ECR for custom training & inference containers
- Lambda for inference API wrapper
- EventBridge Scheduler
- CloudWatch Logs
- CodePipeline + CodeBuild for CI/CD
- IAM Roles (No tokens, no secrets exposed)


🏗 System Architecture Diagram
- Data → Processing → Training → Evaluation → Registry → Deployment → Monitoring

                ┌──────────────────────┐
                │     AWS S3 Bucket    │
                │  Raw / Processed Data│
                └─────────┬────────────┘
                          │
                 (1) Data Ingestion
                          │
                ┌─────────▼───────────┐
                │SageMaker Processing │
                │ Preprocessing / FE   │
                └─────────┬───────────┘
                          │
                      (2) Training
                          │
                ┌─────────▼───────────┐
                │ SageMaker Training  │
                │   (SKLearn / PyTorch)│
                └─────────┬───────────┘
                          │
                     (3) Evaluation
                          │
                ┌─────────▼───────────┐
                │   Model Registry    │
                │ Auto Versioning     │
                └─────────┬───────────┘
                          │
                     (4) Deployment
                          │
                ┌─────────▼───────────┐
                │Real-time Endpoint   │
                └─────────┬───────────┘
                          │
                  (5) Monitoring
                          │
                ┌─────────▼───────────┐
                │ CloudWatch + Monitor│
                └─────────────────────┘

🧪 Pipeline Stages
- Data Ingestion (S3 → Processing Job)
    ✔ Reads raw dataset
    ✔ Writes to processing output

- Data Preprocessing
    ✔ Missing values
    ✔ Normalization
    ✔ Cleanup

- Feature Engineering
    ✔ Feature extraction
    ✔ Create ML-ready features

- Training
    ✔ Train model
    ✔ Store artifacts in S3

- Evaluation
    ✔ Calculate accuracy
    ✔ Save metrics

- Model Registration
    ✔ Auto-version model
    ✔ Store in Model Registry

- Conditional Deployment
    ✔ If metrics meet threshold → deploy else → stop



🎯 Project Goal: 
Build a fully automated MLOps pipeline using only AWS services, where:

- Data is stored in S3
- Processing happens using SageMaker Processing Jobs
- Training is executed using SageMaker Training Jobs
- Model is versioned in SageMaker Model Registry
- Pipelines orchestrate end-to-end ML workflow
- Deployment creates a SageMaker Endpoint
- CI/CD automatically updates models using CodePipeline





🛠 AWS Services Used
| Task                      | AWS Service Used                       |
| ------------------------- | -------------------------------------- |
| Data storage              | **Amazon S3**                          |
| Automated preprocessing   | **SageMaker Processing Jobs**          |
| Model training            | **SageMaker Training Jobs**            |
| Feature engineering       | **SageMaker Processing Jobs**          |
| Model evaluation          | **SageMaker Processing / Training**    |
| Model registry            | **SageMaker Model Registry**           |
| CI/CD for ML              | **SageMaker Pipelines + CodePipeline** |
| Deployment (real-time)    | **SageMaker Endpoints**                |
| Deployment (batch)        | **SageMaker Batch Transform**          |
| Monitoring                | **SageMaker Model Monitor**            |
| Logging                   | **CloudWatch Logs**                    |
| Scheduling retraining     | **EventBridge Cron**                   |
| Docker image hosting      | **Amazon ECR**                         |
| IAM-based security        | **IAM Roles & Policies**               |
| Serverless inference APIs | **AWS Lambda + API Gateway**           |

Scripts:

1. Ingestion

✔ Reads data from s3://bucket/data/sourcedata/
✔ Cleans, splits, writes to data/raw/
✔ Logs → logs/data_ingestion/

2. Preprocessing

✔ Reads raw from S3
✔ Cleans text → writes to data/processed/
✔ Logs → logs/data_preprocessing/

3. Feature Engineering

✔ Loads processed
✔ TF-IDF vectorization
✔ Writes to data/features/
✔ Logs → logs/feature_engineering/

4. Model Training

✔ Loads features
✔ Trains Logistic Regression
✔ Saves model → models/logreg_model.pkl
✔ Saves train metrics → metrics/train_metrics.json
✔ Logs → logs/model_train/

5. Model Evaluation

✔ Loads model using joblib
✔ Loads test features
✔ Computes eval metrics
✔ Saves metrics → evaluation/eval_metrics.json
✔ Logs → logs/model_evaluation/



IAM Roles:

SageMakerExecutionRole

Attach these AWS managed policies:

    AmazonSageMakerFullAccess

    AmazonS3FullAccess

    AmazonEC2ContainerRegistryFullAccess

    CloudWatchLogsFullAccess


SageMakerPipelineRole
