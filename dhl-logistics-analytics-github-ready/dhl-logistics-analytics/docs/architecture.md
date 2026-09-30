# System Architecture

The supplied capstone describes an end-to-end flow:

**S3 ingestion → Airflow ETL → Great Expectations quality gate → Scikit-Learn/XGBoost → MLflow registry → FastAPI `/predict` → SHAP/LIME/DiceML → Fairlearn audit → operations dashboard.**

The operational prediction point is after inbound scanning at the final destination sorting hub and before last-mile loading.

The intended intervention window is approximately 2–6 hours before dispatch.

This repository implements the application-facing components locally. Cloud infrastructure integrations can be connected when authorized AWS, Airflow, MLflow, PostgreSQL, and monitoring environments are available.
