from typing import NamedTuple

from kfp.dsl import Dataset, Input, Output, component


@component(
    base_image="python:3.10", packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def generate_data(dataset: Output[Dataset]):
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=50000, n_features=1000, n_informative=100, n_classes=20
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    df.to_parquet(dataset.path)


@component(base_image="python:3.10", packages_to_install=["pandas", "pyarrow"])
def validate_data(
    dataset: Input[Dataset],
    validated_dataset: Output[Dataset],
    imbalance_threshold: float = 0.02,
) -> NamedTuple("outputs", [("validated", bool)]):
    import logging

    import pandas as pd

    df = pd.read_parquet(dataset.path)

    logging.info("Performing completeness check...")
    columns_with_nans = df.columns[df.isna().any()].tolist()
    if columns_with_nans:
        logging.info(f"⚠️ Missing values detected in columns: {columns_with_nans}")
        return outputs(False)

    logging.info("Performing duplicate check...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logging.info(f"⚠️ {duplicates} duplicates found")
        return outputs(False)

    logging.info("Checking severe class imbalance...")
    class_counts = df.iloc[:, -1].value_counts(normalize=True)
    min_class_freq = class_counts.min()
    if min_class_freq < imbalance_threshold:
        logging.info(
            f"⚠️ Severe imbalance detected. Minority class with {min_class_freq:.4f} ratio"
        )
        return outputs(False)

    logging.info("✅ Dataset validated!")
    df.to_parquet(validated_dataset.path)

    outputs = NamedTuple("outputs", [("validated", bool)])
    return outputs(True)
