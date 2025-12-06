from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.estimator import Estimator
import sagemaker

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
IMAGE = "975248345240.dkr.ecr.us-east-1.amazonaws.com/mlops/mlmodels:latest"
ROLE = "arn:aws:iam::975248345240:role/SageMakerExecutionRole"
BUCKET = "aws-sagemaker-end2end-project"
REGION = "us-east-1"

session = sagemaker.Session()


# -------------------------------------------------------------------
# PIPELINE DEFINITION
# -------------------------------------------------------------------
def get_pipeline():

    # ============================
    # 1️⃣ INGESTION STEP
    # ============================
    ingest_processor = ScriptProcessor(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.t3.medium",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )

    ingest_step = ProcessingStep(
        name="IngestionStep",
        processor=ingest_processor,
        code="src/data/data_ingestion.py"
    )


    # ============================
    # 2️⃣ PREPROCESSING STEP
    # ============================
    preprocess_processor = ScriptProcessor(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.t3.medium",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )

    preprocess_step = ProcessingStep(
        name="PreprocessingStep",
        processor=preprocess_processor,
        code="src/data/data_preprocessing.py",
        depends_on=[ingest_step],
    )


    # ============================
    # 3️⃣ FEATURE ENGINEERING STEP
    # ============================
    feature_processor = ScriptProcessor(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.t3.medium",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )

    feature_step = ProcessingStep(
        name="FeatureEngineeringStep",
        processor=feature_processor,
        code="src/data/feature_engineering.py",
        depends_on=[preprocess_step],
    )


    # ============================
    # 4️⃣ TRAINING STEP
    # ============================
    train_estimator = Estimator(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.m5.large",
        instance_count=1,
        entry_point="src/model/model_train.py",
        source_dir=".",
        output_path=f"s3://{BUCKET}/models/",
        sagemaker_session=session,
    )

    train_step = TrainingStep(
        name="TrainingStep",
        estimator=train_estimator,
        inputs={
            "train": f"s3://{BUCKET}/data/features/"
        },
        depends_on=[feature_step],
    )


    # ============================
    # 5️⃣ EVALUATION STEP
    # ============================
    eval_processor = ScriptProcessor(
        image_uri=IMAGE,
        role=ROLE,
        instance_type="ml.t3.medium",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )

    eval_step = ProcessingStep(
        name="EvaluationStep",
        processor=eval_processor,
        code="src/model/model_evaluation.py",
        depends_on=[train_step],
    )


    # ============================
    # FINAL PIPELINE ASSEMBLY
    # ============================
    pipeline = Pipeline(
        name="MLOpsEndToEndPipeline",
        steps=[
            ingest_step,
            preprocess_step,
            feature_step,
            train_step,
            eval_step,
        ],
        sagemaker_session=session,
    )

    return pipeline


# -------------------------------------------------------------------
# PIPELINE EXECUTION ENTRYPOINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print("Pipeline execution started:", execution.arn)
