import os
from typing import NamedTuple

from kfp.dsl import Dataset, Input, Output, component

pipeline_name = os.getenv("PIPELINE_NAME")

@component(
    base_image="europe-docker.pkg.dev/rnd-data-ml/base-images/cpu-mlflow-py3.12",
    packages_to_install=["pandas", "pyarrow"]
)
def validate_data(
    dataset: Input[Dataset],
    validated_dataset: Output[Dataset],
    imbalance_threshold: float = 0.02,
) -> NamedTuple("outputs", [("validated", bool)]):
    import logging

    import pandas as pd

    outputs = NamedTuple("outputs", [("validated", bool)])

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

    return outputs(True)

