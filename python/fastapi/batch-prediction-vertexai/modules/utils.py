import numpy as np
from typing import List, Dict, Any, Optional, Union

def process_prediction_probabilities(
    pred_probs: Union[np.ndarray, List[List[float]]],
    n: int = 3,
    class_labels: Optional[Dict[int, Any]] = None,
    output_format: str = 'dict'
) -> List[Dict[str, List[Any]]]:
    """
    Process prediction probabilities to extract top N predictions with confidence scores.
    
    Parameters:
    -----------
    pred_probs : numpy.ndarray or List[List[float]]
        Probability distribution from a classification model's predict_proba method
    
    n : int, optional (default=3)
        Number of top predictions to return
    
    class_labels : dict, optional
        Mapping of indices to class names. If None, uses index as label
    
    output_format : str, optional (default='dict')
        Format of the output. Options:
        - 'dict': Returns a list of dictionaries with 'labels' and 'scores'
        - 'tuple': Returns a list of tuples with (label, score)
    
    Returns:
    --------
    List of predictions, each containing top N predictions and their confidence scores
    
    Examples:
    ---------
    >>> import numpy as np
    >>> pred_probs = np.array([[0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    >>> class_labels = {0: 'cat', 1: 'dog', 2: 'bird'}
    >>> process_prediction_probabilities(pred_probs, class_labels=class_labels)
    [
        {
            'labels': ['dog', 'bird', 'cat'],
            'scores': [0.6, 0.3, 0.1]
        },
        {
            'labels': ['bird', 'dog', 'cat'],
            'scores': [0.5, 0.3, 0.2]
        }
    ]
    """
    pred_probs = np.asarray(pred_probs)
    if pred_probs.ndim != 2:
        raise ValueError("❗ Input must be a 2D array of probabilities")

    if class_labels is None:
        class_labels = {i: i for i in range(pred_probs.shape[1])}
    
    top_n_idx = np.argsort(pred_probs, axis=1)[:, -n:][:, ::-1]
    
    results = []
    for i in range(pred_probs.shape[0]):
        top_indeces = top_n_idx[i].tolist()
        labels = [class_labels.get(idx, idx) for idx in top_indeces]
        confidences = pred_probs[i, top_indeces].tolist()
        
        if output_format == 'dict':
            prediction = {
                'labels': labels,
                'scores': confidences
            }
        elif output_format == 'tuple':
            prediction = list(zip(labels, confidences))
        else:
            raise ValueError(f"❓ Unsupported output format: {output_format}")
        
        results.append(prediction)
    
    return results

def process_batch_predictions(
    instances: List[Union[str, List[str]]],
    prediction_func: Callable[[List[str]], List[Dict[str, Any]]],
    id_extractor: Optional[Callable[[Union[str, List[str]]], str]] = None,
    default_prediction: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generic batch prediction processor with flexible input and output handling.
    
    Parameters:
    -----------
    instances : List[Union[str, List[str]]]
        Input instances to predict. Can be strings or lists of strings.
    
    prediction_func : Callable[[List[str]], List[Dict[str, Any]]]
        Function that takes a list of input IDs and returns predictions.
        Typically a function that does the actual prediction logic.
    
    id_extractor : Optional[Callable[[Union[str, List[str]]], str]]
        Optional function to extract ID from an instance.
        If None, uses a default extraction strategy.
    
    default_prediction : Optional[Dict[str, Any]]
        Optional default prediction to use if no prediction is available.
    
    Returns:
    --------
    List[Dict[str, Any]]
        Batch predictions aligned with input instances.
    
    Examples:
    ---------
    >>> def sample_prediction_func(resource_ids):
    ...     return [
    ...         {"topics": ["Topic1"], "confidence_scores": [0.9]},
    ...         {"topics": ["Topic2"], "confidence_scores": [0.8]}
    ...     ]
    >>> 
    >>> instances = ["id1", ["id2"], "id3"]
    >>> process_batch_predictions(instances, sample_prediction_func)
    [
        {"topics": ["Topic1"], "confidence_scores": [0.9]},
        {"topics": ["Topic2"], "confidence_scores": [0.8]},
        {"topics": [], "confidence_scores": []}
    ]
    """
    if id_extractor is None:
        def id_extractor(instance):
            if isinstance(instance, list) and len(instance) > 0:
                return str(instance[0])
            return str(instance)
    
    instances = [id_extractor(instance) for instance in instances]
    
    raw_predictions = prediction_func(instances)
    
    predictions = []
    for i, _ in enumerate(instances):
        if i < len(raw_predictions):
            prediction = {
                "topics": raw_predictions[i].get("topics", []),
                "confidence_scores": raw_predictions[i].get("confidence_scores", [])
            }
        else:
            prediction = default_prediction or {
                "topics": [],
                "confidence_scores": []
            }
        predictions.append(prediction)
    
    return predictions
