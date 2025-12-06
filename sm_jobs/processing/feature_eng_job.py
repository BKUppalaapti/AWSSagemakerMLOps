from sagemaker.processing import ScriptProcessor
import sagemaker

IMAGE = "975248345240.dkr.ecr.us-east-1.amazonaws.com/mlops/mlmodels:latest"
ROLE = "arn:aws:iam::975248345240:role/SageMakerExecutionRole"
session = sagemaker.Session()

def run_feature_eng():
    print("Starting Feature Engineering Job...")

    processor = ScriptProcessor(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.t3.medium",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )

    processor.run(code="src/data/feature_engineering.py")

    print("Feature Engineering job submitted successfully.")

if __name__ == "__main__":
    run_feature_eng()
