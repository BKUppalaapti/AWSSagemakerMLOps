import sagemaker
from sagemaker.estimator import Estimator

IMAGE = "975248345240.dkr.ecr.us-east-1.amazonaws.com/mlops/mlmodels:latest"
ROLE = "arn:aws:iam::975248345240:role/SageMakerExecutionRole"
BUCKET = "aws-sagemaker-end2end-project"

session = sagemaker.Session()

def run_train():
    print("Starting Model Training Job...")

    estimator = Estimator(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.m5.large",
        instance_count=1,
        entry_point="src/model/model_train.py",  # YOUR SCRIPT
        source_dir=".",                           # because src/ is copied to /opt/ml/code
        output_path=f"s3://{BUCKET}/models/",
        sagemaker_session=session,
    )

    estimator.fit({
        "train": f"s3://{BUCKET}/data/features/"
    })

    print("Training job submitted successfully.")

if __name__ == "__main__":
    run_train()
