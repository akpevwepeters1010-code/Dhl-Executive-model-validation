# Operational Governance

## Human oversight

Predictions are advisory. Dispatchers validate recommendations against capacity, cost, safety, service and operational constraints.

## Concept drift

The capstone specifies weekly drift testing and investigation/retraining when ROC-AUC falls below 0.88.

## Fairness

Fairness should be evaluated across HUB_LOCATION and SERVICE_LEVEL. Recommended monitored metrics include:

- False Negative Rate
- False Positive Rate
- Selection Rate
- Equalized Odds disparity

The capstone describes Fairlearn ExponentiatedGradient with an Equalized Odds constraint as a mitigation approach.

## Data quality

The described pipeline uses:

1. SHA-256 anonymization
2. median imputation
3. outlier capping
4. Great Expectations quality gates
5. separation of raw and model-ready data

## Auditability

Preserve:

- prediction logs
- model versions
- feature versions
- validation results
- fairness audit results
- ground-truth reconciliation

No personally identifiable information should be committed to Git.
