import os

from kfp.dsl import Dataset, Input, Output, component

pipeline_name = os.getenv("PIPELINE_NAME")


@component(
    base_image="europe-docker.pkg.dev/rnd-data-ml/base-images/cpu-mlflow-py3.12",
    packages_to_install=["pandas", "pyarrow", "mlflow", "scikit-learn"],
)
def preprocess_data(
    dataset: Input[Dataset],
    train_set: Output[Dataset],
    val_set: Output[Dataset],
    test_set: Output[Dataset],
    test_size: float = 0.3,
):
    import logging

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(dataset.path)
    X, y = df.drop("target", axis=1), df["target"]

    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns, index=X.index
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    df_train = X_train.assign(target=y_train)
    df_val = X_val.assign(target=y_val)
    df_test = X_test.assign(target=y_test)

    logging.info("🧱 Data successfully split and logged.")

    for data, output in zip(
        [df_train, df_val, df_test], [train_set, val_set, test_set]
    ):
        data.to_parquet(output.path)

    logging.info(f"Train set shape: {df_train.shape}")
    logging.info(f"Validation set shape: {df_val.shape}")
    logging.info(f"Test set shape: {df_test.shape}")
