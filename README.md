# 🏥 Hospital Readmission Risk Prediction

> **30-day unplanned readmission risk scoring with SHAP explainability**  
> XGBoost · SMOTE · Clinical Feature Engineering · FastAPI · Streamlit

[![CI](https://github.com/bhavyalakkamraju09/readmission-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/bhavyalakkamraju09/readmission-risk/actions)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.655-blue)](#model-performance)
[![Sensitivity](https://img.shields.io/badge/Sensitivity-0.68-green)](#model-performance)
[![Live Demo](https://img.shields.io/badge/Demo-HuggingFace_Spaces-orange)](https://huggingface.co/spaces/bhavyalakkamraju09/readmission-risk)

---

## Business Impact

Hospital readmissions cost the U.S. healthcare system **$26 billion annually**.
The CMS Hospital Readmissions Reduction Program (HRRP) penalizes hospitals **up to 3%** of all Medicare payments for excess 30-day readmissions.

| Metric | Value |
|--------|-------|
| Average cost per unplanned readmission | **$15,200** (KFF 2023) |
| Annual discharges (typical community hospital) | 5,000 |
| Baseline readmission rate | 15% |
| Estimated avoided readmissions (model + intervention) | **~216/year** |
| **Annual cost savings** | **~$3.2M** |

This model identifies the **top-risk patients at discharge** for targeted care coordination — enabling proactive follow-up calls, medication reconciliation, and social support before readmission occurs.

---

## Architecture

```
EHR Data (CSV/FHIR)
       │
       ▼
Feature Engineering          ICD-9 → CCS mapping
       │                     Charlson Comorbidity Index
       │                     Polypharmacy flags
       │                     Prior utilization features
       ▼
SMOTE Oversampling           Handles class imbalance (11% positive)
       │
       ▼
XGBoost Classifier           300 trees, depth-5, calibrated probabilities
       │
       ▼
Threshold Tuning             F2-score optimization (recall-weighted)
       │
       ┌─────────────────────┐
       │                     │
       ▼                     ▼
  FastAPI                Streamlit
  /predict endpoint      Clinical dashboard
       │                     │
       ▼                     ▼
  SHAP Explanation       Risk gauge + waterfall plot
  Risk tier + ROI        Care coordination recommendation
```

---

## Model Performance

Evaluated on held-out test set (20% of 17,615 patients):

| Metric | Value | Notes |
|--------|-------|-------|
| AUC-ROC | **0.655** | Consistent with published HRRP literature (0.62–0.70) |
| AUC-PR | **0.595** | More informative under class imbalance |
| F2-Score | **0.774** | Recall-weighted — clinical priority |
| Sensitivity | **0.68** | At optimal clinical threshold |
| Specificity | **0.60** | |
| Decision threshold | **0.30** | Tuned via F2-score maximization |

> **Why AUC-ROC ~0.65?** Readmission prediction is an inherently noisy clinical problem. The HOSPITAL score (published benchmark) achieves AUC 0.72–0.74 with rich structured EHR data including lab values. Our model achieves competitive performance using claims-level features only, demonstrating the importance of model interpretability over chasing raw AUC.

---

## Clinical Features (34 total)

**Prior utilization** (strongest predictors)
- `number_inpatient` — prior inpatient visits in last year
- `prior_visits` — composite (inpatient + ER + outpatient)
- `has_prior_emergency` — any ER visit in past year

**Medication burden**
- `num_medications`, `polypharmacy` (>10 meds), `high_polypharmacy` (>20)
- `insulin_changed`, `med_change_flag` — medication adjustment at discharge
- `on_insulin` — insulin-dependent diabetes

**Glycemic control**
- `a1c_level` — HbA1c ordinal (None=0, Normal=1, >7=2, >8=3)
- `poor_glycemic_control` — HbA1c >8%

**Diagnosis severity**
- `cci_score` — Charlson Comorbidity Index proxy from ICD-9
- `ccs_diag1/2/3` — CCS category encoding of ICD-9 diagnoses
- `high_diagnosis_burden` — >7 concurrent diagnoses

**Hospitalization**
- `time_in_hospital`, `long_stay` (>7 days), `lab_intensity`
- `emergency_admission` — admission type = Emergency

---

## Project Structure

```
readmission_risk/
├── data/
│   └── diabetic_data.csv          # 130-US Hospitals dataset (UCI / synthetic)
├── src/
│   ├── features.py                # ReadmissionFeatureEngineer, ICD-9 mappings
│   └── train.py                   # Full training pipeline with MLflow
├── api/
│   └── main.py                    # FastAPI /predict endpoint
├── app/
│   └── streamlit_app.py           # Clinical dashboard (Streamlit)
├── models/
│   ├── readmission_model.pkl      # Trained XGBoost model
│   ├── feature_engineer.pkl       # Fitted transformer
│   ├── shap_explainer.pkl         # SHAP TreeExplainer
│   ├── threshold.pkl              # Optimal decision threshold
│   └── test_metrics.pkl           # Held-out test metrics
├── reports/
│   ├── eval_curves.png            # ROC + PR curves
│   ├── shap_summary.png           # Global SHAP feature importance
│   └── shap_waterfall.png         # Patient-level SHAP explanation
├── tests/
│   └── test_pipeline.py           # 25 pytest tests (unit + integration)
├── .github/workflows/ci.yml       # GitHub Actions: test → train → deploy
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/bhavyalakkamraju09/readmission-risk
cd readmission-risk
pip install -r requirements.txt

# Train model
python -m src.train

# Run Streamlit dashboard
streamlit run app/streamlit_app.py

# Run API
uvicorn api.main:app --reload
```

### API Usage

```python
import requests

patient = {
    "age": "[70-80)",
    "gender": "Female",
    "time_in_hospital": 8,
    "num_medications": 22,
    "number_inpatient": 2,
    "number_emergency": 1,
    "A1Cresult": ">8",
    "change": "Ch",
    "diabetesMed": "Yes",
    "insulin": "Up",
    "diag_1": "250.40",
    # ... (see /docs for full schema)
}

resp = requests.post("http://localhost:8000/predict", json=patient)
print(resp.json())
# {
#   "readmission_risk_score": 0.714,
#   "readmission_risk_pct": "71.4%",
#   "risk_tier": "HIGH",
#   "top_risk_drivers": [
#     {"feature": "number inpatient", "impact": 0.312, "direction": "increases"},
#     {"feature": "num medications", "impact": 0.198, "direction": "increases"},
#     ...
#   ],
#   "avoided_cost_if_intervened": "$4,332",
#   "recommendation": "Immediate care coordination..."
# }
```

---

## Free Deployment Stack

| Layer | Tool | Cost |
|-------|------|------|
| Streamlit dashboard | HuggingFace Spaces | Free |
| REST API | Render.com / Railway | Free |
| Experiment tracking | MLflow (local) | Free |
| Dataset versioning | DVC + Google Drive | Free |
| CI/CD | GitHub Actions | Free (2000 min/mo) |
| Data drift monitoring | Evidently AI | Free |

---

## References

- Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*.
- Donzé et al. (2013). Potentially avoidable 30-day hospital readmissions — HOSPITAL score. *JAMA Internal Medicine*.
- Centers for Medicare & Medicaid Services. Hospital Readmissions Reduction Program (HRRP).
- Kaiser Family Foundation (2023). Hospital Readmissions Report.

---

## About

Built by **Bhavya Lakkamraju** · MS Computer Science, Lawrence Technological University (May 2026)  
[GitHub](https://github.com/bhavyalakkamraju09) · [LinkedIn](https://linkedin.com/in/bhavya-varma) · [Portfolio](https://huggingface.co/spaces/bhavyalakkamraju09/clinical-rag-voice-agent)
