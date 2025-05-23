from kfp.dsl import Dataset, Input, component


@component(
    base_image="europe-docker.pkg.dev/rnd-data-ml/base-images/cpu-mlflow-py3.12",
    packages_to_install=[
        "pandas",
        "mlflow",
        "google-cloud-storage",
        "matplotlib",
    ],
)
def evaluate_model(
    experiment_name: str,
    run_id_1: str,
    run_id_2: str,
    test_set: Input[Dataset],
):
    import logging

    import mlflow
    import pandas as pd
    from mlflow_utils import initialize_mlflow

    logging.basicConfig(level=logging.INFO)

    try:
        logging.info("📡 Connecting to MLflow Server...")
        initialize_mlflow(experiment_name=experiment_name)

        logging.info("📥 Loading test dataset...")
        test_df = pd.read_parquet(test_set.path)

        logging.info("🛒 Collecting training artifacts...")
        run_ids = [run_id_1, run_id_2]

        logging.info(f"🧪 Evaluating {len(run_ids)} model(s)...")
        for i, run_id in enumerate(run_ids, start=1):
            try:
                logging.info(f"📦 Loading model #{i}")
                with mlflow.start_run(run_id=run_id):
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.pyfunc.load_model(model_uri)

                    mlflow.evaluate(
                        model=model,
                        data=test_df,
                        targets="target",
                        model_type="classifier",
                    )
                logging.info(f"👍 Successfully logged evaluation metrics for model #{i}")
            except Exception as e:
                raise RuntimeError(f"Failed to evaluate model #{i}: {e}")

        logging.info("🎉 All model evaluations completed!")

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise RuntimeError(f"Model evaluation failed: {e}")
