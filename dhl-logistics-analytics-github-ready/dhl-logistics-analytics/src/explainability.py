"""Explainability helpers for trained pipelines."""
import joblib
import pandas as pd

def explain_with_shap(model_path, X):
    import shap
    model = joblib.load(model_path)
    transformed = model.named_steps["preprocess"].transform(X)
    estimator = model.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    return explainer(transformed)

def explain_with_lime(model_path, training_data, row):
    from lime.lime_tabular import LimeTabularExplainer
    model = joblib.load(model_path)
    pre = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    X_train = pre.transform(training_data)
    explainer = LimeTabularExplainer(
        X_train.toarray() if hasattr(X_train, "toarray") else X_train,
        mode="classification"
    )
    transformed = pre.transform(row)
    return explainer.explain_instance(
        transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0],
        estimator.predict_proba
    )
