"""Fairness auditing utilities using Fairlearn."""
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, false_positive_rate, false_negative_rate,
    selection_rate
)

def audit(y_true, y_pred, sensitive_features):
    frame = MetricFrame(
        metrics={
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return frame.by_group
