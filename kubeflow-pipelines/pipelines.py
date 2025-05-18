import os

from kfp import local
from kfp.client.client import compiler
from kfp.dsl import pipeline

from components.data import generate_data, validate_data
from components.prep import preprocess_data
from components.train import train_rfc, train_svc, train_xgb
from components.evaluate import evaluate_model

local.init(local.DockerRunner())

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

@pipeline()
def pipeline_op(experiment_name: str, mlflow_uri: str = mlflow_uri):
    data_op = generate_data()
    data_validation_op = validate_data(dataset=data_op.outputs["dataset"])
    if data_validation_op.outputs["validated"] == True:
    # with If(data_validation_op.outputs["Output"] == True, name="check_validation"):
        prep_data_op = preprocess_data(
            dataset=data_validation_op.outputs["validated_dataset"]
        )
        train_svc_op = train_svc(
            train_set=prep_data_op.outputs["train_set"],
            val_set=prep_data_op.outputs["val_set"],
            mlflow_uri=mlflow_uri,
            experiment_name=experiment_name,
        )
        # train_rfc_op = train_rfc(
        #     train_set=prep_data_op.outputs["train_set"],
        #     val_set=prep_data_op.outputs["val_set"],
        #     mlflow_uri=mlflow_uri,
        #     experiment_name=experiment_name,
        # )
        model_eval_op = evaluate_model(
            model_1=train_svc_op.outputs["model"],
            run_id_1=train_svc_op.outputs["run_id"],
            test_set=prep_data_op.outputs["test_set"],
            mlflow_uri=mlflow_uri,
            experiment_name=experiment_name,
        )
        # model_val_op
        # model_select_op


if __name__ == "__main__":
    # compiler.Compiler().compile(pipeline_op, "simple-pipeline.yaml")
    pipeline_op(experiment_name="Simple Pipeline Experiments")
