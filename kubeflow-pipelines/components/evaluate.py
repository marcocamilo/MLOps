from typing import Optional
from kfp.dsl import component, Input, Dataset, Model

@component(
    base_image="python:3.10", 
    target_image="gcr.io/your-project/kfp/txc/evaluation-component:latest",
    packages_to_install=["pandas", "joblib", "mlflow"],
)
def evaluate_model(
    model_1: Input[Model],
    run_id_1: str,
    test_set: Input[Dataset],
    mlflow_uri: str,
    experiment_name: str,
):
    import logging
    import joblib
    import mlflow
    import pandas as pd
    from mlflow_utils import setup_mlflow

    logging.basicConfig(level=logging.INFO)

    logging.info("📡 Connecting to MLflow Server...")
    setup_mlflow(mlflow_uri, experiment_name)

    logging.info("📥 Loading test dataset, models and run_ids...")
    test_df = pd.read_parquet(test_set.path)

    training_metadata = []

    if model_1 and run_id_1:
        training_metadata.append({"model": model_1.path, "run_id": run_id_1})
    # if model_2 and run_id_2:
    #     training_metadata.append({"model": model_2.path, "run_id": run_id_2})
    # if model_3 and run_id_3:
    #     training_metadata.append({"model": model_3.path, "run_id": run_id_3})

    if not training_metadata:
        raise ValueError("❌ No valid model/run_id pairs provided for evaluation.")

    logging.info(f"🧪 Evaluating {len(training_metadata)} model(s)...")

    for i, metadata in enumerate(training_metadata, start=1):
        logging.info(f"\n📦 Loading model #{i} from: {metadata['model']}")
        model = joblib.load(metadata["model"])

        logging.info(f"🔍 Evaluating model #{i} under run ID: {metadata['run_id']}")
        with mlflow.start_run(run_id=metadata["run_id"]):
            mlflow.evaluate(
                model=model,
                data=test_df,
                targets="target",
                model_type="classifier",
            )

    logging.info("🎉 All model evaluations completed successfully!")
