import os

from google.oauth2 import service_account
from kfp.dsl import pipeline, If
from kfp.client.client import compiler
from kfp.registry import RegistryClient

from src.generate import generate_data
from src.validate import validate_data
from src.preprocess import preprocess_data
from src.train import train_svc, train_lr
from src.evaluate import evaluate_model

from google.cloud import aiplatform

@pipeline
def ml_pipeline(
    experiment_name: str = "KFP - Kubeflow Pipelines",
) -> None:
    """
   End-to-end ML pipeline that generates data, validates it, trains challenger models, and evaluates performance.
   
   Pipeline flow:
   1. Generate synthetic dataset
   2. Validate data quality and schema
   3. Preprocess data (train/val/test splits)
   4. Train challenger models (currently SVC)
   5. Evaluate all models against test set
   
   Args:
       experiment_name: MLflow experiment name for tracking runs and models
    """
    data_op = generate_data()
    data_validation_op = validate_data(dataset=data_op.outputs["dataset"])

    with If(data_validation_op.outputs["validated"] == True, name="check_validation"):
        prep_data_op = preprocess_data(
            dataset=data_validation_op.outputs["validated_dataset"]
        )

        train_svc_op = train_svc(
            train_set=prep_data_op.outputs["train_set"],
            val_set=prep_data_op.outputs["val_set"],
            experiment_name=experiment_name,
        )

        train_svc_op_2 = train_svc(
            train_set=prep_data_op.outputs["train_set"],
            val_set=prep_data_op.outputs["val_set"],
            experiment_name=experiment_name,
        )

        evaluate_op = evaluate_model(
            experiment_name=experiment_name,
            run_id_1=train_svc_op.outputs["run_id"],
            run_id_2=train_svc_op_2.outputs["run_id"],
            test_set=prep_data_op.outputs["test_set"]
        )

if __name__ == "__main__":
    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION")
    zone = os.getenv("ZONE")
    pipeline_name = os.getenv("PIPELINE_NAME")
    repository = os.getenv("REPOSITORY")
    host = os.getenv("HOST")
    service_account = os.getenv("SERVICE_ACCOUNT")
    staging_bucket = os.getenv("STAGING_BUCKET")

    pipeline_file = f"{pipeline_name}.yaml"
    compiler.Compiler().compile(ml_pipeline, pipeline_file)

    client = RegistryClient(host=host)
    template_name, version_name = client.upload_pipeline(
        file_name=pipeline_file,
        tags=["latest"],
        extra_headers={
            "description": "ML Pipeline with MLflow tracking that works on Vertex AI."
        },
    )

    template_path = f"https://{region}-kfp.pkg.dev/{project_id}/{repository}/{template_name}/{version_name}"

    aiplatform.init(
        project=project_id,
        location=zone,
        staging_bucket=repository,
    )
    pipeline_job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=template_path,
        pipeline_root=staging_bucket,
        enable_caching=False,
        failure_policy="fast"
    )
    pipeline_job.submit(
        service_account=service_account,
    )
