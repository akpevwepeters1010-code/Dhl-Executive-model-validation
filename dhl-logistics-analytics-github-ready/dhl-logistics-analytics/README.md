# DHL Logistics Analytics — SLA Breach Early-Warning System

> BAN6800 Business Analytics Capstone — Executive Decision-Support & Technical Governance

This repository implements the technical architecture described in the BAN6800 capstone report for an early-warning machine-learning system that predicts shipment SLA-breach risk approximately **2–6 hours before last-mile dispatch**.

## Project objective

The system is designed as a **human-in-the-loop decision-support mechanism**, not an autonomous replacement for dispatch personnel.

The reported capstone results are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Naive baseline | 0.820 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.845 | 0.612 | 0.584 | 0.598 | 0.782 |
| Random Forest | 0.912 | 0.785 | 0.812 | 0.798 | 0.914 |
| XGBoost | **0.941** | **0.842** | **0.894** | **0.867** | **0.958** |

The repository contains reproducible implementation code, but the original 5,500-row DHL enterprise dataset and trained model artifact were **not included with the supplied report**. Do not represent generated/demo data as DHL production data.

## Architecture

```text
Raw CSV / S3
     |
     v
Airflow ETL
     |
     v
SHA-256 PII anonymization
     |
     v
Imputation + outlier capping
     |
     v
Great Expectations quality gate
     |
     v
PostgreSQL / feature dataset
     |
     v
Scikit-learn + XGBoost
     |
     v
MLflow model tracking
     |
     +--------------------+
     |                    |
     v                    v
FastAPI /predict       SHAP / LIME / DiceML
     |                    |
     +---------+----------+
               v
         Fairlearn audit
               |
               v
   Streamlit / Operations Dashboard
               |
               v
 Prometheus + Evidently + human review
```

## Repository structure

```text
dhl-logistics-analytics/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── governance.md
│   └── model_card.md
├── models/
├── notebooks/
├── src/
│   ├── data_pipeline.py
│   ├── explainability.py
│   ├── fairness.py
│   ├── train.py
│   └── synthetic_data.py
├── tests/
│   └── test_api.py
├── config/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Features

- Binary SLA-breach classification
- XGBoost training pipeline
- Class-imbalance handling with `scale_pos_weight`
- Probability threshold of `0.40` for operational warning
- CRITICAL / ELEVATED / LOW risk categories
- SHAP explainability
- LIME local explanations
- Fairlearn fairness auditing
- FastAPI prediction service
- Streamlit operations dashboard
- Pydantic input validation
- Docker deployment
- Prometheus-compatible metrics endpoint
- Drift/governance documentation
- Unit tests

## Risk categories

- **CRITICAL:** probability ≥ 0.75
- **ELEVATED:** probability ≥ 0.40 and < 0.75
- **LOW:** probability < 0.40

These thresholds reproduce the operating logic described in the report.

## Input features

The report identifies the following operational variables:

- `hub_location`
- `service_level`
- `transit_duration`
- `weather_severity`
- `traffic_density`
- `route_distance`
- `package_weight`

The training code also accepts a shipment identifier and anonymized customer identifier for data lineage; these identifiers are not used as predictive features.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate clearly labeled demo data

```bash
python src/synthetic_data.py
```

This creates a synthetic dataset for testing the software pipeline. It is **not DHL enterprise data**.

### 4. Train a demo model

```bash
python src/train.py --input data/raw/demo_shipments.csv --output models/xgboost_pipeline.joblib
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/docs
```

### 6. Start the dashboard

```bash
streamlit run dashboard/app.py
```

## API example

POST `/predict`

```json
{
  "hub_location": "HUB_US",
  "service_level": "Express_Worldwide",
  "transit_duration": 18.5,
  "weather_severity": 3,
  "traffic_density": 7,
  "route_distance": 850.0,
  "package_weight": 12.4
}
```

Response:

```json
{
  "delay_probability": 0.82,
  "prediction": "Delayed",
  "risk_category": "CRITICAL",
  "recommended_action": "Escalate to priority courier and consider dynamic rerouting."
}
```

## Governance

The report specifies:

- human confirmation for model recommendations
- weekly concept-drift tests
- retraining/investigation when ROC-AUC falls below 0.88
- monthly Fairlearn fairness audits
- Pydantic validation and fallback routing
- preservation of prediction logs and model versions
- monitoring of Precision, Recall and F1 after ground-truth reconciliation

See `docs/governance.md`.

## Important data note

The supplied capstone report states that the analytical dataset contains 5,500 historical shipment records and reports model metrics from that work. The dataset itself was not supplied with the document used to build this repository. Therefore:

1. The repository does not claim that its demo model reproduces the reported metrics.
2. No real customer information should be committed.
3. The demo dataset is synthetic and exists only to validate the application.
4. To reproduce the academic results, place the authorized source dataset in `data/raw/` and run the training pipeline.

## Author

**Akpevwe Peters**  
BAN6800 — Business Analytics  
Nexford University (NXU)

## Reference

The repository is based on the supplied BAN6800 Business Analytics Capstone Report, whose stated project repository is:

`github.com/akpevwepeters1010-code/dhl-logistics-analytics`
