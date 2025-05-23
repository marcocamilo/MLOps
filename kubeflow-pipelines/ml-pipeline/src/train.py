from typing import NamedTuple

from kfp.dsl import Dataset, Input, Model, Output, component


def train_component():
    return component(
        base_image="europe-docker.pkg.dev/rnd-data-ml/base-images/cpu-mlflow-py3.12",
        packages_to_install=[
            "pandas",
            "scikit-learn",
            "optuna",
            "mlflow",
            "xgboost",
            "google-cloud-storage",
        ],
    )


@train_component()
def train_svc(
    experiment_name: str,
    train_set: Input[Dataset],
    val_set: Input[Dataset],
    n_trials: int = 3,
) -> NamedTuple("outputs", [("run_id", str)]):
    import logging
    from datetime import datetime

    import mlflow
    import optuna
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow_utils import get_run_id, initialize_mlflow
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    from sklearn.svm import LinearSVC

    try:
        logging.info("📡 Connecting to MLflow Server...")
        initialize_mlflow(experiment_name=experiment_name)

        logging.info("📥 Loading training and validation data...")
        df_train = pd.read_parquet(train_set.path)
        df_val = pd.read_parquet(val_set.path)

        X_train = df_train.drop("target", axis=1).values
        y_train = df_train["target"].values
        X_val = df_val.drop("target", axis=1).values
        y_val = df_val["target"].values

        train_dataset = mlflow.data.from_pandas(
            df_train, targets="target", name="training_set"
        )
        val_dataset = mlflow.data.from_pandas(
            df_val, targets="target", name="validation_set"
        )

        def objective(trial):
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)
            max_iter = trial.suggest_int("max_iter", 500, 2000)

            try:
                clf = LinearSVC(C=C, max_iter=max_iter, random_state=24)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                score = accuracy_score(y_val, y_pred)

                logging.info(f"🤖 Performing experiment for trial {trial.number}")
                with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                    mlflow.log_params(
                        {
                            "C": C,
                            "max_iter": max_iter,
                        }
                    )
                    mlflow.log_metric("accuracy_score", score)

                    metrics = {
                        "accuracy_score": accuracy_score(y_val, y_pred),
                        "precision_score": precision_score(
                            y_val, y_pred, average="weighted"
                        ),
                        "recall_score": recall_score(y_val, y_pred, average="weighted"),
                        "f1_score": f1_score(y_val, y_pred, average="weighted"),
                    }
                    mlflow.log_metrics(metrics)

                return score

            except Exception as e:
                logging.warning(f"Trial {trial.number} failed: {e}")
                return 0.0

        logging.info("⚙️ Setting run name...")
        run_name = f"LinearSVC-{datetime.now().strftime('%y%m%d-%H%M%S')}"
        run_id, run_name = get_run_id(run_name, experiment_name)

        logging.info("🚀 Starting MLflow run...")
        with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
            mlflow.set_tags(
                {"estimator_name": "LinearSVC", "stage": "model_development"}
            )

            logging.info(
                f"🎯 Starting hyperparameter tuning with Optuna ({n_trials} trials)..."
            )
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            logging.info(f"✅ Best parameters found: {study.best_params}")
            best_params = study.best_params
            best_model = LinearSVC(**best_params, random_state=24)
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            logging.info(f"🏆 Final model accuracy: {acc:.4f}")

            metrics = {
                "accuracy_score": accuracy_score(y_val, y_pred),
                "precision_score": precision_score(y_val, y_pred, average="weighted"),
                "recall_score": recall_score(y_val, y_pred, average="weighted"),
                "f1_score": f1_score(y_val, y_pred, average="weighted"),
            }
            mlflow.log_metrics(metrics)

            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(val_dataset, context="testing")

            signature = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_val[:5],
            )

            logging.info("💾 Saving model artifact...")
            logging.info("✅ Model saved!")

            final_run_id = run.info.run_id

        outputs = NamedTuple("outputs", [("run_id", str)])
        return outputs(final_run_id)
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise RuntimeError(f"Model training failed: {e}")


@train_component()
def train_lr(
    experiment_name: str,
    train_set: Input[Dataset],
    val_set: Input[Dataset],
    model: Output[Model],
    n_trials: int = 15,
) -> NamedTuple("outputs", [("model_uri", str)]):
    import logging
    from datetime import datetime

    import mlflow
    import optuna
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow_utils import get_run_id, initialize_mlflow
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)

    try:
        logging.info("📡 Connecting to MLflow Server...")
        initialize_mlflow(experiment_name=experiment_name)

        logging.info("📥 Loading training and validation data...")
        df_train = pd.read_parquet(train_set.path)
        df_val = pd.read_parquet(val_set.path)

        X_train = df_train.drop("target", axis=1).values
        y_train = df_train["target"].values
        X_val = df_val.drop("target", axis=1).values
        y_val = df_val["target"].values

        train_dataset = mlflow.data.from_pandas(
            df_train, targets="target", name="training_set"
        )
        val_dataset = mlflow.data.from_pandas(
            df_val, targets="target", name="validation_set"
        )

        def objective(trial):
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
            penalty = trial.suggest_categorical(
                "penalty", ["l1", "l2", "elasticnet", None]
            )

            # Handle solver-penalty compatibility
            if solver == "liblinear" and penalty == "elasticnet":
                penalty = "l2"
            elif solver == "lbfgs" and penalty in ["l1", "elasticnet"]:
                penalty = "l2"
            elif penalty == "elasticnet":
                solver = "saga"

            max_iter = trial.suggest_int("max_iter", 100, 1000)

            try:
                clf = LogisticRegression(
                    C=C,
                    solver=solver,
                    penalty=penalty,
                    max_iter=max_iter,
                    random_state=24,
                    n_jobs=-1,
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                score = accuracy_score(y_val, y_pred)

                logging.info(f"🤖 Performing experiment for trial {trial.number}")
                with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                    mlflow.log_params(
                        {
                            "C": C,
                            "solver": solver,
                            "penalty": penalty,
                            "max_iter": max_iter,
                        }
                    )

                    metrics = {
                        "accuracy_score": accuracy_score(y_val, y_pred),
                        "precision_score": precision_score(
                            y_val, y_pred, average="weighted"
                        ),
                        "recall_score": recall_score(y_val, y_pred, average="weighted"),
                        "f1_score": f1_score(y_val, y_pred, average="weighted"),
                    }
                    mlflow.log_metrics(metrics)

                return score

            except Exception as e:
                logging.warning(f"Trial {trial.number} failed: {e}")
                return 0.0

        logging.info("⚙️ Setting run name...")
        run_name = f"LogisticRegression-{datetime.now().strftime('%y%m%d-%H%M%S')}"
        run_id, run_name = get_run_id(run_name, experiment_name)

        logging.info("🚀 Starting MLflow run...")
        with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
            mlflow.set_tags(
                {"estimator_name": "LogisticRegression", "stage": "model_development"}
            )

            logging.info(
                f"🎯 Starting hyperparameter tuning with Optuna ({n_trials} trials)..."
            )
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            if study.best_value == 0.0:
                raise ValueError("All trials failed during hyperparameter optimization")

            logging.info(f"✅ Best parameters found: {study.best_params}")
            best_params = study.best_params
            best_model = LogisticRegression(**best_params, random_state=24, n_jobs=-1)
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            logging.info(f"🏆 Final model validation accuracy: {acc:.4f}")

            metrics = {
                "accuracy_score": accuracy_score(y_val, y_pred),
                "precision_score": precision_score(y_val, y_pred, average="weighted"),
                "recall_score": recall_score(y_val, y_pred, average="weighted"),
                "f1_score": f1_score(y_val, y_pred, average="weighted"),
            }
            mlflow.log_metrics(metrics)

            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(val_dataset, context="validation")

            signature = infer_signature(X_val, y_pred)
            model_info = mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_val[:5],
            )

            logging.info("💾 Saving model artifact...")
            logging.info("✅ Model saved!")

            model_uri = model_info.model_uri

        outputs = NamedTuple("outputs", [("model_uri", str)])
        return outputs(model_uri)

    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise RuntimeError(f"Model training failed: {e}")
