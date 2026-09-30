"""Train the SLA-breach classification pipeline."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

FEATURES = [
    "hub_location", "service_level", "transit_duration",
    "weather_severity", "traffic_density", "route_distance",
    "package_weight"
]
TARGET = "delayed"

def build_pipeline():
    categorical = ["hub_location", "service_level"]
    numeric = [c for c in FEATURES if c not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric)
    ])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=4.5,
        eval_metric="logloss",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])

def main(input_path, output_path):
    df = pd.read_csv(input_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.40).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, p),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, output_path)
    print("Model saved:", output_path)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="models/xgboost_pipeline.joblib")
    args = parser.parse_args()
    main(args.input, args.output)
