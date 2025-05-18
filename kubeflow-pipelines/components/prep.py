from kfp.dsl import Dataset, Input, Output, component


@component(
    base_image="python:3.10", packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def preprocess_data(
    dataset: Input[Dataset],
    train_set: Output[Dataset],
    val_set: Output[Dataset],
    test_set: Output[Dataset],
    test_size: float = 0.3
):
    import logging
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(dataset.path)
    X = df.drop("target", axis=1).values
    y = df["target"].values
    X_cols = [f"f{i}" for i in range(X.shape[1])]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    df_train = pd.DataFrame(X_train, columns=X_cols)
    df_train["target"] = y_train

    df_val = pd.DataFrame(X_val, columns=X_cols)
    df_val["target"] = y_val

    df_test = pd.DataFrame(X_test, columns=X_cols)
    df_test["target"] = y_test

    for data, output in zip(
        [df_train, df_val, df_test], [train_set, val_set, test_set]
    ):
        data.to_parquet(output.path)

    logging.info(f"Train set shape: {df_train.shape}")
    logging.info(f"Validation set shape: {df_val.shape}")
    logging.info(f"Test set shape: {df_test.shape}")
