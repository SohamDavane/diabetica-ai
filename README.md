# 🩺 DiabéticaAI — Commercial-Grade Diabetes Risk Platform

> **Enterprise ML Platform** | XGBoost + LightGBM + Random Forest Ensemble | SHAP Explainability | FastAPI | HIPAA-Conscious

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)]()

---

## Architecture Overview

```
diabetica/
├── ml/
│   └── pipeline.py          # Data loading, preprocessing, ensemble training, SHAP
├── api/
│   └── main.py              # FastAPI backend with rate limiting, audit logging
├── monitoring/
│   └── drift.py             # PSI + KS + JS data drift detection engine
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# With Pima dataset (baseline):
python -m ml.pipeline

# With CDC BRFSS dataset (recommended for production):
python -c "
from ml.pipeline import build_and_train_pipeline
build_and_train_pipeline(
    pima_path='data/pima.csv',
    brfss_path='data/diabetes_indicator.csv'
)
"
```

### 3. Start the API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test a Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 3,
    "Glucose": 148.0,
    "BloodPressure": 72.0,
    "SkinThickness": 35.0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50.0,
    "patient_id": "PT-001"
  }'
```

---

## API Endpoints

| Method | Endpoint             | Description                              |
|--------|----------------------|------------------------------------------|
| GET    | `/health`            | Liveness/readiness probe                 |
| GET    | `/model/metrics`     | Training & cross-validation metrics      |
| POST   | `/predict`           | Single patient risk assessment           |
| POST   | `/predict/batch`     | Batch scoring (up to 100 patients)       |
| GET    | `/risk/tiers`        | Risk tier definitions & recommendations  |
| GET    | `/schema/features`   | Feature schema for EHR integration       |
| GET    | `/docs`              | Interactive Swagger UI                   |

---

## Model Performance Targets

| Metric        | Target  | Clinical Rationale                              |
|---------------|---------|--------------------------------------------------|
| F1-Score      | > 0.85  | Balance precision and sensitivity                |
| Recall        | > 0.90  | Minimize false negatives (missed diagnoses)      |
| ROC-AUC       | > 0.92  | Strong discrimination across risk thresholds     |
| 10-fold CV F1 | > 0.83  | Generalization assurance                         |

---

## Data Sources

### 1. Pima Indians Diabetes Dataset (baseline)
- 768 records, 8 features
- Source: UCI ML Repository

### 2. CDC BRFSS Diabetes Health Indicators (recommended)
- 253,680 records, 21 features
- Source: [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Annual CDC survey, nationally representative

### 3. Custom EHR Integration
- Use `DataLoader.load_custom_ehr(path, target_col)`
- Automatic schema harmonization

---

## Risk Stratification

| Tier     | Probability | Follow-up  | Action                        |
|----------|-------------|------------|-------------------------------|
| Low      | 0–30%       | 12 months  | Lifestyle maintenance         |
| Moderate | 30–60%      | 3 months   | DPP enrollment + dietitian    |
| High     | 60–80%      | 2 weeks    | Endocrinology referral        |
| Critical | 80–100%     | 48–72 hrs  | Urgent clinical evaluation    |

---

## SHAP Explainability

Each prediction returns a `clinical_narrative` and ranked `top_risk_factors`:

```json
{
  "clinical_narrative": "Clinical risk factors identified: Primary drivers — Blood Glucose Level (148.0); Body Mass Index (33.6). Contributing factors — Patient Age. Protective factors — Physical Activity.",
  "top_risk_factors": [
    {"label": "Blood Glucose Level", "value": 148.0, "shap_value": 0.42, "direction": "increases", "impact": "High"},
    {"label": "Body Mass Index",     "value": 33.6,  "shap_value": 0.28, "direction": "increases", "impact": "High"}
  ]
}
```

---

## Data Drift Monitoring

Run weekly drift detection against incoming EHR data:

```python
from monitoring.drift import FeatureDriftDetector, DriftReporter

detector = FeatureDriftDetector(reference_data=training_df)
report   = detector.detect(current_week_df)
# → Saves JSON report, triggers alerts if PSI > 0.10
```

**Thresholds:**
- PSI < 0.10 → Stable
- PSI 0.10–0.25 → Investigate
- PSI > 0.25 → Retrain model

---

## Docker Deployment

```bash
# Build
docker build -t diabetica-api:1.0.0 .

# Run
docker run -d \
  -p 8000:8000 \
  -e API_SECRET_KEY=<secret> \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  diabetica-api:1.0.0
```

---

## Compliance Notes

- **HIPAA**: Patient IDs are SHA-256 hashed in audit logs. No PHI stored in prediction responses.
- **GDPR Article 22**: SHAP explanations provide the "meaningful information about the logic involved" required for automated decisions.
- **FDA SaMD**: Positioned as Clinical Decision Support (CDS) — not a medical device. Physician review required.
- **Audit Trail**: Full request/response logging at `logs/audit.log`

---

## Roadmap

- [ ] HL7 FHIR R4 native resource output
- [ ] Federated learning for multi-hospital training without data sharing
- [ ] Real-time streaming inference via Apache Kafka
- [ ] Automated retraining DAG (Apache Airflow)
- [ ] CE Mark / FDA 510(k) pre-submission preparation

---

*DiabéticaAI © 2025 — Proprietary. All rights reserved.*
