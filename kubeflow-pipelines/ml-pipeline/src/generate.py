import os
from typing import NamedTuple

from kfp.dsl import Dataset, Input, Output, component

pipeline_name = os.getenv("PIPELINE_NAME")

@component(
    base_image="europe-docker.pkg.dev/rnd-data-ml/base-images/cpu-mlflow-py3.12",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def generate_data(dataset: Output[Dataset]):
    import logging
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=100, n_informative=10, n_classes=20
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    df.to_parquet(dataset.path)
    logging.info("⚙️ Data successfully generated")
