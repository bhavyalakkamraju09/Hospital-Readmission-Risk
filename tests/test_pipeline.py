"""
Tests for hospital readmission risk pipeline.
Run: pytest tests/ -v
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np
import joblib

from src.features import (
    ReadmissionFeatureEngineer,
    age_to_midpoint,
    map_icd9_to_ccs,
    compute_cci_proxy,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ── Helper fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def sample_patient():
    return {
        'age': '[60-70)', 'gender': 'Female', 'race': 'Caucasian',
        'time_in_hospital': 8, 'num_lab_procedures': 55,
        'num_procedures': 2, 'num_medications': 22,
        'number_outpatient': 0, 'number_emergency': 1,
        'number_inpatient': 2, 'number_diagnoses': 8,
        'diag_1': '250.00', 'diag_2': '428.0', 'diag_3': '?',
        'admission_type_id': 1, 'discharge_disposition_id': 1,
        'admission_source_id': 4,
        'A1Cresult': '>8', 'change': 'Ch', 'diabetesMed': 'Yes',
        'insulin': 'Up', 'metformin': 'Steady', 'repaglinide': 'No',
        'nateglinide': 'No', 'glimepiride': 'No', 'glipizide': 'No',
        'glyburide': 'No', 'weight': '?', 'payer_code': 'MC',
        'medical_specialty': 'InternalMedicine',
    }


@pytest.fixture
def low_risk_patient():
    return {
        'age': '[30-40)', 'gender': 'Male', 'race': 'Caucasian',
        'time_in_hospital': 2, 'num_lab_procedures': 20,
        'num_procedures': 0, 'num_medications': 5,
        'number_outpatient': 0, 'number_emergency': 0,
        'number_inpatient': 0, 'number_diagnoses': 2,
        'diag_1': '250.00', 'diag_2': '?', 'diag_3': '?',
        'admission_type_id': 3, 'discharge_disposition_id': 1,
        'admission_source_id': 1,
        'A1Cresult': 'Norm', 'change': 'No', 'diabetesMed': 'No',
        'insulin': 'No', 'metformin': 'No', 'repaglinide': 'No',
        'nateglinide': 'No', 'glimepiride': 'No', 'glipizide': 'No',
        'glyburide': 'No', 'weight': '?', 'payer_code': 'MC',
        'medical_specialty': 'Family/GeneralPractice',
    }


@pytest.fixture
def engineer():
    return ReadmissionFeatureEngineer()


# ── Unit tests: feature functions ──────────────────────────────────────────────
class TestFeatureFunctions:
    def test_age_midpoint_standard(self):
        assert age_to_midpoint('[60-70)') == 65

    def test_age_midpoint_young(self):
        assert age_to_midpoint('[0-10)') == 5

    def test_age_midpoint_old(self):
        assert age_to_midpoint('[90-100)') == 95

    def test_age_midpoint_nan(self):
        result = age_to_midpoint(None)
        assert isinstance(result, int)

    def test_icd9_ccs_diabetes(self):
        assert map_icd9_to_ccs('250.00') == 49

    def test_icd9_ccs_cardio(self):
        assert map_icd9_to_ccs('428.0') == 108

    def test_icd9_ccs_unknown(self):
        assert map_icd9_to_ccs('999.99') == 0

    def test_icd9_ccs_missing(self):
        assert map_icd9_to_ccs('?') == 0

    def test_cci_diabetes(self):
        score = compute_cci_proxy('250.00')
        assert score == 1

    def test_cci_renal(self):
        score = compute_cci_proxy('585.3')
        assert score == 2

    def test_cci_unknown(self):
        assert compute_cci_proxy('?') == 0


# ── Integration tests: feature engineering ─────────────────────────────────────
class TestFeatureEngineer:
    def test_output_shape(self, engineer, sample_patient):
        X = engineer.transform(pd.DataFrame([sample_patient]))
        assert X.shape[0] == 1
        assert X.shape[1] > 20

    def test_no_nulls(self, engineer, sample_patient):
        X = engineer.transform(pd.DataFrame([sample_patient]))
        assert X.isnull().sum().sum() == 0, "Features contain NaN values"

    def test_expected_columns_present(self, engineer, sample_patient):
        X = engineer.transform(pd.DataFrame([sample_patient]))
        required = ['age_num', 'polypharmacy', 'cci_score', 'prior_visits',
                    'a1c_level', 'lab_intensity', 'emergency_admission']
        for col in required:
            assert col in X.columns, f"Missing column: {col}"

    def test_polypharmacy_flag(self, engineer, sample_patient):
        sample_patient['num_medications'] = 22
        X = engineer.transform(pd.DataFrame([sample_patient]))
        assert X['polypharmacy'].values[0] == 1

    def test_no_polypharmacy(self, engineer, sample_patient):
        sample_patient['num_medications'] = 5
        X = engineer.transform(pd.DataFrame([sample_patient]))
        assert X['polypharmacy'].values[0] == 0

    def test_long_stay_flag(self, engineer, sample_patient):
        sample_patient['time_in_hospital'] = 10
        X = engineer.transform(pd.DataFrame([sample_patient]))
        assert X['long_stay'].values[0] == 1

    def test_batch_transform(self, engineer, sample_patient, low_risk_patient):
        df = pd.DataFrame([sample_patient, low_risk_patient])
        X = engineer.transform(df)
        assert X.shape[0] == 2

    def test_prior_visits_sum(self, engineer, sample_patient):
        sample_patient['number_inpatient'] = 2
        sample_patient['number_emergency'] = 1
        sample_patient['number_outpatient'] = 0
        X = engineer.transform(pd.DataFrame([sample_patient]))
        assert X['prior_visits'].values[0] == 3


# ── Integration tests: trained model ──────────────────────────────────────────
@pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, 'readmission_model.pkl')),
    reason="Model not trained yet"
)
class TestModelPrediction:
    @pytest.fixture
    def model(self):
        return joblib.load(os.path.join(MODEL_DIR, 'readmission_model.pkl'))

    @pytest.fixture
    def eng(self):
        return joblib.load(os.path.join(MODEL_DIR, 'feature_engineer.pkl'))

    def test_predict_probability_range(self, model, eng, sample_patient):
        X = eng.transform(pd.DataFrame([sample_patient]))
        prob = model.predict_proba(X.values)[0][1]
        assert 0.0 <= prob <= 1.0

    def test_high_risk_patient_higher_than_low(self, model, eng,
                                               sample_patient, low_risk_patient):
        X_high = eng.transform(pd.DataFrame([sample_patient]))
        X_low  = eng.transform(pd.DataFrame([low_risk_patient]))
        prob_high = model.predict_proba(X_high.values)[0][1]
        prob_low  = model.predict_proba(X_low.values)[0][1]
        assert prob_high >= prob_low, (
            f"High-risk patient scored lower ({prob_high:.3f}) than low-risk ({prob_low:.3f})")

    def test_feature_names_match(self, model, eng, sample_patient):
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        X = eng.transform(pd.DataFrame([sample_patient]))
        assert list(X.columns) == feature_names

    def test_metrics_above_baseline(self):
        metrics = joblib.load(os.path.join(MODEL_DIR, 'test_metrics.pkl'))
        assert metrics['auc_roc'] > 0.55, "AUC-ROC should beat naive baseline"
        assert metrics['sensitivity'] > 0.50, "Sensitivity too low for clinical use"


# ── Business logic tests ───────────────────────────────────────────────────────
class TestBusinessLogic:
    def test_roi_calculation(self):
        """Basic ROI calculation: avoided readmissions × cost."""
        prob = 0.75
        cost_per_readmission = 15200
        intervention_success = 0.40
        avoided_cost = prob * cost_per_readmission * intervention_success
        assert avoided_cost == pytest.approx(4560.0)

    def test_risk_tier_thresholds(self):
        """Risk tier assignment matches expected boundaries."""
        def get_tier(prob):
            if prob >= 0.60: return "HIGH"
            elif prob >= 0.38: return "MODERATE"
            else: return "LOW"

        assert get_tier(0.80) == "HIGH"
        assert get_tier(0.50) == "MODERATE"
        assert get_tier(0.20) == "LOW"
        assert get_tier(0.60) == "HIGH"
        assert get_tier(0.38) == "MODERATE"
