from sagemaker.processing import ScriptProcessor
import sagemaker

IMAGE = "975248345240.dkr.ecr.us-east-1.amazonaws.com/mlops/mlmodels:latest"
ROLE = "arn:aws:iam::975248345240:role/SageMakerExecutionRole"
session = sagemaker.Session()

def run_eval():
    print("Starting Evaluation Job...")

    processor = ScriptProcessor(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.t3.medium",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )

    processor.run(code="src/model/model_evaluation.py")

    print("Evaluation job submitted successfully.")

if __name__ == "__main__":
    run_eval()
