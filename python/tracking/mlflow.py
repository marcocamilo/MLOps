import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
from typing import Optional, Any, Tuple, Dict

def get_run_id(run_name, experiment_name=experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}"'
    )
    return runs[0].info.run_id if runs else None, run_name

def log_mlflow_run(
    run_name: str,
    stage: str,
    model: Optional[Any] = None,
    datasets: Optional[Dict[str, Tuple[Any, Any]]] = None,
    metrics: Optional[Dict[str, float]] = None,
    plots: Optional[Dict[str, plt.Figure]] = None,
    save_model: bool = False,
    model_io: Optional[Tuple[Any, Any]] = None,
    artifact_name: str = "model"
):
    stages = {
        "preprocessing": "prep",
        "feature_engineering": "fe",
        "model_development": "md",
        "model_testing": "test",
        "model_evaluation": "eval",
    }
    if stage not in stages:
        available_stages = ", ".join(sorted(stages.keys()))
        raise ValueError(
            f"Invalid stage: '{stage}'. "
            f"Available stages are: {available_stages}"
        )
    stage_code = stages[stage]    
    run_id, run_name = get_run_id(run_name)
    with mlflow.start_run(run_id=run_id, run_name=run_name):
        
        # Tags
        if model is not None:
            mlflow.set_tags({
                "estimator_name": model.__class__.__name__,
                "estimator_class": f"{model.__class__.__module__}",
                "stage": stage
            })
        
        # Datasets
        if datasets is not None:
            for label, data in datasets.items():
                context = None
                if label in ["validation", "training"]:
                    context = "testing"
                else:
                    context = "training"
                # Numpy split
                mlflow_dataset = mlflow.data.from_numpy(
                    features=data[0], 
                    targets=data[1],
                    name=f"{context}-data-{stage_code}"
                )
                mlflow.log_input(mlflow_dataset, context=context)
        
        # Metrics
        if metrics is not None:
            mlflow.log_metrics(metrics)
        
        # Visualizations
        if plots is not None:
            for name, plot in plots.items():
                mlflow.log_figure(plot, f"{name}.png")
        
        # Log model with signature only if save_model is True
        if save_model and model is not None:
            try:
                # Ensure model_io is provided
                if model_io is None:
                    raise ValueError("model_io must be provided when save_model is True")
                
                signature = infer_signature(model_io[0], model_io[1])
                
                mlflow.sklearn.log_model(
                    model, 
                    artifact_name,
                    signature=signature,
                    input_example=model_io[0][:5],
                )
            except Exception as e:
                print(f"Could not log model signature: {e}")
                mlflow.sklearn.log_model(model, artifact_name)
