"""
Feature engineering for hospital readmission risk prediction.
Maps ICD-9 codes → CCS groups, computes Charlson Comorbidity Index proxy,
and builds clinical risk features.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ICD-9 → CCS (Clinical Classification Software) group mapping
ICD9_CCS_MAP = {
    # Diabetes
    '250.00': 49, '250.01': 49, '250.02': 49, '250.10': 49,
    '250.11': 49, '250.40': 49, '250.60': 49,
    # Cardiovascular
    '414.01': 101, '414.00': 101, '428.0': 108, '427.31': 106,
    '401.9': 98, '410.01': 100, '412': 100,
    # Respiratory
    '486': 122, '490': 127, '491.21': 127, '496': 127,
    '518.81': 131, '507.0': 122, '482.9': 122,
    # Renal
    '585.3': 158, '585.4': 158, '585.5': 158,
    '403.91': 158, '584.9': 157, '588.1': 158,
    # Other
    '276.1': 55, '276.51': 55, '285.9': 59,
    '272.4': 53, '311': 657, 'V58.61': 49, '038.9': 2,
}

# Charlson Comorbidity Index weights by ICD-9 prefix
CCI_WEIGHTS = {
    '410': 1, '411': 1, '412': 1, '414': 1,  # MI
    '428': 1,                                  # CHF
    '440': 1, '441': 1,                        # PVD
    '430': 1, '431': 1, '432': 1, '433': 1, '434': 1, '436': 1,  # CVD
    '290': 1,                                  # Dementia
    '490': 1, '491': 1, '492': 1, '496': 1,   # COPD
    '710': 1, '714': 1, '725': 1,             # Rheumatic
    '531': 1, '532': 1, '533': 1, '534': 1,   # PUD
    '571': 1, '573': 1,                        # Mild liver
    '250': 1,                                  # Diabetes uncomplicated
    '342': 2, '343': 2, '344': 2,             # Hemiplegia
    '582': 2, '583': 2, '585': 2, '586': 2,   # Renal
    '140': 2, '141': 2, '172': 2, '174': 2,   # Malignancy
    '042': 6,                                  # HIV
    '197': 6, '198': 6, '199': 6,             # Metastatic cancer
    '572': 3, '456': 3,                        # Severe liver
}


def compute_cci_proxy(diag_code: str) -> int:
    """Estimate CCI weight from primary ICD-9 code prefix."""
    if pd.isna(diag_code) or diag_code == '?':
        return 0
    prefix = str(diag_code)[:3]
    return CCI_WEIGHTS.get(prefix, 0)


def map_icd9_to_ccs(diag_code: str) -> int:
    """Map ICD-9 code to CCS category."""
    if pd.isna(diag_code) or diag_code == '?':
        return 0
    return ICD9_CCS_MAP.get(str(diag_code).strip(), 0)


def age_to_midpoint(age_str: str) -> int:
    """Convert age bracket '[50-60)' → 55."""
    if pd.isna(age_str):
        return 45
    try:
        low = int(age_str.strip('[').split('-')[0])
        return low + 5
    except Exception:
        return 45


def encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    """Encode medication change columns as ordinal."""
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'glimepiride',
                'glipizide', 'glyburide', 'insulin']
    med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}
    for col in med_cols:
        if col in df.columns:
            df[f'{col}_enc'] = df[col].map(med_map).fillna(0)
    return df


class ReadmissionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Full feature engineering pipeline for readmission risk.
    Produces clinically-meaningful features from raw EHR columns.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # --- Age ---
        df['age_num'] = df['age'].apply(age_to_midpoint)

        # --- ICD-9 mappings ---
        df['ccs_diag1'] = df['diag_1'].apply(map_icd9_to_ccs)
        df['ccs_diag2'] = df['diag_2'].apply(map_icd9_to_ccs)
        df['ccs_diag3'] = df['diag_3'].apply(map_icd9_to_ccs)

        # --- Charlson Comorbidity Index (proxy from primary dx) ---
        df['cci_score'] = (
            df['diag_1'].apply(compute_cci_proxy) +
            df['diag_2'].apply(compute_cci_proxy) * 0.5 +
            df['diag_3'].apply(compute_cci_proxy) * 0.3
        ).clip(0, 10)

        # --- Polypharmacy ---
        df['polypharmacy'] = (df['num_medications'] > 10).astype(int)
        df['high_polypharmacy'] = (df['num_medications'] > 20).astype(int)

        # --- Lab intensity ---
        df['lab_intensity'] = (df['num_lab_procedures'] /
                               df['time_in_hospital'].clip(1, None))

        # --- Prior utilization ---
        df['prior_visits'] = (df['number_inpatient'] +
                              df['number_outpatient'] +
                              df['number_emergency'])
        df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
        df['has_prior_emergency'] = (df['number_emergency'] > 0).astype(int)

        # --- Medication context ---
        df['med_change_flag'] = ((df['change'] == 'Ch') &
                                  (df['diabetesMed'] == 'Yes')).astype(int)
        df['on_insulin'] = df['insulin'].apply(
            lambda x: 0 if x == 'No' else 1)
        df['insulin_changed'] = df['insulin'].apply(
            lambda x: 1 if x in ['Up', 'Down'] else 0)

        # --- Glycemic control ---
        a1c_map = {'>8': 3, '>7': 2, 'Norm': 1, 'None': 0}
        df['a1c_level'] = df['A1Cresult'].map(a1c_map).fillna(0)
        df['poor_glycemic_control'] = (df['a1c_level'] == 3).astype(int)

        # --- LOS risk ---
        df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
        df['very_long_stay'] = (df['time_in_hospital'] > 10).astype(int)

        # --- Emergency admission ---
        df['emergency_admission'] = (df['admission_type_id'] == 1).astype(int)

        # --- Diagnosis burden ---
        df['high_diagnosis_burden'] = (df['number_diagnoses'] > 7).astype(int)

        # --- Gender ---
        df['gender_enc'] = (df['gender'] == 'Male').astype(int)

        # --- Specialist care flag ---
        specialist_specs = {'Cardiology', 'Nephrology', 'Pulmonology', 'Psychiatry'}
        df['specialist_care'] = df['medical_specialty'].apply(
            lambda x: 1 if x in specialist_specs else 0)

        # --- Medication encodings ---
        df = encode_medications(df)

        # Select final feature columns
        feature_cols = [
            'age_num', 'gender_enc',
            'time_in_hospital', 'long_stay', 'very_long_stay',
            'num_lab_procedures', 'num_procedures', 'num_medications',
            'lab_intensity', 'polypharmacy', 'high_polypharmacy',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'prior_visits', 'has_prior_inpatient', 'has_prior_emergency',
            'number_diagnoses', 'high_diagnosis_burden',
            'ccs_diag1', 'ccs_diag2', 'ccs_diag3',
            'cci_score',
            'a1c_level', 'poor_glycemic_control',
            'med_change_flag', 'on_insulin', 'insulin_changed',
            'emergency_admission', 'specialist_care',
            'metformin_enc', 'glipizide_enc', 'glyburide_enc', 'insulin_enc',
        ]
        return df[[c for c in feature_cols if c in df.columns]]

    def get_feature_names_out(self, input_features=None):
        return self.transform(pd.DataFrame(columns=input_features or [])).columns.tolist()
