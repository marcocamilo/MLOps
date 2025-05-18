from typing import Tuple

import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(mlflow_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)


def get_run_id(
    run_name: str, experiment_name: str = "Default"
) -> Tuple[str, str]:
    """
    Check if a run with the given name exists in the specified MLflow experiment.
    If it exists, return its run_id; otherwise, return (run_name, None).

    Parameters:
    ----------
    run_name : str
        Name of the MLflow run (can be treated as a tag or attribute).
    experiment_name : str
        Name of the experiment to search in. Defaults to "Default".

    Returns:
    -------
    Tuple[str, Optional[str]]
        A tuple of the run ID if found, else None, and run name.
    """
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )

    if runs:
        return runs[0].info.run_id, run_name
    return None, run_name
