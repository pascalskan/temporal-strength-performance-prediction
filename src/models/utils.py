from sklearn.pipeline import Pipeline

def get_model_attribute(model, attribute_name):
    """
    Safely retrieves an attribute from a model, whether it's a plain
    estimator or a scikit-learn Pipeline.

    If the model is a Pipeline, it attempts to retrieve the attribute from the
    final step of the pipeline.

    Args:
        model: The model or Pipeline object.
        attribute_name (str): The name of the attribute to retrieve (e.g., "coef_").

    Returns:
        The attribute value if found, otherwise None.
    """
    if isinstance(model, Pipeline):
        # Access the final estimator in the pipeline
        final_estimator = model.steps[-1][1]
        return getattr(final_estimator, attribute_name, None)
    else:
        # Direct access for non-pipeline models
        return getattr(model, attribute_name, None)
