from typing import NamedTuple

from kfp.dsl import Dataset, Input, Model, Output, component


def train_component():
    return component(
        base_image="python:3.10",
        target_image="gcr.io/your-project/kfp/txc/train-component:latest",
        packages_to_install=["pandas", "scikit-learn", "joblib", "optuna", "mlflow", "xgboost"],
    )


@train_component()
def train_svc(
    train_set: Input[Dataset],
    val_set: Input[Dataset],
    mlflow_uri: str,
    experiment_name: str,
    model: Output[Model],
) -> NamedTuple("outputs", [("run_id", str)]):
    import logging

    import joblib
    import mlflow
    import optuna
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow_utils import setup_mlflow, get_run_id
    from sklearn.metrics import accuracy_score
    from sklearn.svm import LinearSVC

    logging.basicConfig(level=logging.INFO)

    logging.info("📡 Connecting to MLflow Server...")
    setup_mlflow(mlflow_uri, experiment_name)

    logging.info("📥 Loading training and validation data...")
    df_train = pd.read_parquet(train_set.path)
    df_val = pd.read_parquet(val_set.path)

    X_train = df_train.drop("target", axis=1).values
    y_train = df_train["target"].values
    X_val = df_val.drop("target", axis=1).values
    y_val = df_val["target"].values

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        max_iter = trial.suggest_int("max_iter", 500, 2000)
        clf = LinearSVC(C=C, max_iter=max_iter, random_state=24)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        score = accuracy_score(y_val, y_pred)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params({"C": C, "max_iter": max_iter})
            mlflow.log_metric("accuracy_score", score)

        return score

    logging.info("⚙️ Setting run name...")
    run_name = "LinearSVC"
    run_id, run_name = get_run_id(run_name, experiment_name)

    logging.info("🚀 Starting MLflow run...")
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        mlflow.set_tags({"estimator_name": "LinearSVC", "stage": "model_development"})

        logging.info("🎯 Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=3)

        logging.info(f"✅ Best parameters found: {study.best_params}")
        best_params = study.best_params
        best_model = LinearSVC(**best_params, random_state=24)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logging.info(f"🏆 Final model accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy_score", acc)

        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_val[:5],
        )

        logging.info("💾 Saving model artifact...")
        joblib.dump(best_model, model.path)
        logging.info("✅ Model saved!")

        final_run_id = run.info.run_id

    outputs = NamedTuple("outputs", [("run_id", str)])
    return outputs(final_run_id)


@train_component()
def train_rfc(
    train_set: Input[Dataset],
    val_set: Input[Dataset],
    mlflow_uri: str,
    experiment_name: str,
    model: Output[Model],
) -> NamedTuple("outputs", [("run_id", str)]):
    import logging

    import joblib
    import mlflow
    import optuna
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow_utils import setup_mlflow, get_run_id
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    logging.basicConfig(level=logging.INFO)

    logging.info("📡 Connecting to MLflow Server...")
    setup_mlflow(mlflow_uri, experiment_name)

    logging.info("📥 Loading training and validation data...")
    df_train = pd.read_parquet(train_set.path)
    df_val = pd.read_parquet(val_set.path)

    X_train = df_train.drop("target", axis=1).values
    y_train = df_train["target"].values
    X_val = df_val.drop("target", axis=1).values
    y_val = df_val["target"].values

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=24,
            n_jobs=-1,
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        score = accuracy_score(y_val, y_pred)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }
            )
            mlflow.log_metric("accuracy_score", score)

        return score

    logging.info("⚙️ Setting run name...")
    run_name = "RandomForestClassifier"
    run_id, run_name = get_run_id(run_name, experiment_name)

    logging.info("🚀 Starting MLflow run...")
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        mlflow.set_tags(
            {
                "estimator_name": "RandomForestClassifier",
                "stage": "model_development",
            }
        )

        logging.info("🎯 Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)

        logging.info(f"✅ Best parameters found: {study.best_params}")
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params, random_state=24, n_jobs=-1)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logging.info(f"🏆 Final model accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy_score", acc)

        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_val[:5],
        )

        logging.info("💾 Saving model artifact...")
        joblib.dump(best_model, model.path)
        logging.info("✅ Model saved!")

        final_run_id = run.info.run_id

    outputs = NamedTuple("outputs", [("run_id", str)])
    return outputs(final_run_id)


@train_component()
def train_xgb(
    train_set: Input[Dataset],
    val_set: Input[Dataset],
    mlflow_uri: str,
    experiment_name: str,
    model: Output[Model],
) -> NamedTuple("outputs", [("run_id", str)]):
    import logging

    import joblib
    import mlflow
    import optuna
    import pandas as pd
    from mlflow.models import infer_signature
    from mlflow_utils import setup_mlflow, get_run_id
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    logging.basicConfig(level=logging.INFO)

    logging.info("📡 Connecting to MLflow Server...")
    setup_mlflow(mlflow_uri, experiment_name)

    logging.info("📥 Loading training and validation data...")
    df_train = pd.read_parquet(train_set.path)
    df_val = pd.read_parquet(val_set.path)

    X_train = df_train.drop("target", axis=1).values
    y_train = df_train["target"].values
    X_val = df_val.drop("target", axis=1).values
    y_val = df_val["target"].values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        clf = XGBClassifier(
            **params, random_state=24, eval_metric="logloss"
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        score = accuracy_score(y_val, y_pred)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy_score", score)

        return score

    logging.info("⚙️ Setting run name...")
    run_name = "XGBClassifier"
    run_id, run_name = get_run_id(run_name, experiment_name)

    logging.info("🚀 Starting MLflow run...")
    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        mlflow.set_tags(
            {"estimator_name": "XGBClassifier", "stage": "model_development"}
        )

        logging.info("🎯 Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)

        logging.info(f"✅ Best parameters found: {study.best_params}")
        best_params = study.best_params
        best_model = XGBClassifier(
            **best_params,
            random_state=24,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logging.info(f"🏆 Final model accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy_score", acc)

        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_val[:5],
        )

        logging.info("💾 Saving model artifact...")
        joblib.dump(best_model, model.path)
        logging.info("✅ Model saved!")

        final_run_id = run.info.run_id

    outputs = NamedTuple("outputs", [("run_id", str)])
    return outputs(final_run_id)
