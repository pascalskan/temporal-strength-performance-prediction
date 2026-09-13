from sklearn.pipeline import Pipeline


def get_final_estimator(model):
    """
    Returns the final estimator from a sklearn Pipeline,
    otherwise returns the model itself.
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1]

    return model


def get_model_attribute(model, attr):
    """
    Safely retrieves an attribute from either:
    - a raw estimator
    - a Pipeline final estimator
    """
    estimator = get_final_estimator(model)

    if hasattr(estimator, attr):
        return getattr(estimator, attr)

    return None


def supports_feature_importance(model):
    return get_model_attribute(model, "feature_importances_") is not None


def supports_coefficients(model):
    return get_model_attribute(model, "coef_") is not None