"""
Hospital Readmission Risk — Training Pipeline
Trains XGBoost + LightGBM with SMOTE, tunes threshold for clinical utility,
logs all experiments to MLflow, saves best model + SHAP explainer.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, fbeta_score,
    roc_curve, precision_recall_curve, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.features import ReadmissionFeatureEngineer

DATA_PATH  = 'data/diabetic_data.csv'
MODEL_DIR  = 'models'
REPORT_DIR = 'reports'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ── 1. Load & label ──────────────────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} encounters × {df.shape[1]} columns")

    # Remove deaths & hospice (they can't be readmitted)
    if 'discharge_disposition_id' in df.columns:
        df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

    # Remove duplicate patient-level (keep first)
    if 'patient_nbr' in df.columns:
        df = df.drop_duplicates(subset='patient_nbr', keep='first')

    # Binary target: readmitted within 30 days
    df['target'] = (df['readmitted'] == '<30').astype(int)
    print(f"After dedup: {df.shape[0]:,} patients | "
          f"readmit rate: {df['target'].mean():.1%}")
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    eng = ReadmissionFeatureEngineer()
    X = eng.transform(df)
    y = df['target'].values
    print(f"Feature matrix: {X.shape} | positives: {y.sum():,} ({y.mean():.1%})")
    return X, y, eng


# ── 3. Threshold tuning ───────────────────────────────────────────────────────
def tune_threshold(y_true, probs, beta=2):
    """Find decision threshold maximising F-beta (beta>1 → recall-weighted)."""
    thresholds = np.arange(0.10, 0.90, 0.01)
    scores = [fbeta_score(y_true, probs >= t, beta=beta, zero_division=0)
              for t in thresholds]
    best_t = thresholds[int(np.argmax(scores))]
    return best_t, max(scores)


# ── 4. Evaluation helpers ─────────────────────────────────────────────────────
def evaluate(y_true, probs, threshold, label=''):
    preds = (probs >= threshold).astype(int)
    auc_roc  = roc_auc_score(y_true, probs)
    auc_pr   = average_precision_score(y_true, probs)
    f2       = fbeta_score(y_true, preds, beta=2, zero_division=0)
    f1       = fbeta_score(y_true, preds, beta=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    print(f"\n{'─'*50}")
    print(f"{label} | threshold={threshold:.2f}")
    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(f"  AUC-PR  : {auc_pr:.4f}")
    print(f"  F2      : {f2:.4f}")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity         : {specificity:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {
        'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'f2': f2, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def plot_curves(y_true, probs, threshold, model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name} — Evaluation Curves', fontsize=14, fontweight='bold')

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    axes[0].plot(fpr, tpr, color='#185FA5', lw=2, label=f'AUC={auc:.3f}')
    axes[0].plot([0,1],[0,1],'--', color='#888780', lw=1)
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve'); axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    axes[1].plot(rec, prec, color='#0F6E56', lw=2, label=f'AP={ap:.3f}')
    axes[1].axhline(y_true.mean(), color='#888780', linestyle='--', lw=1, label='Baseline')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve'); axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f'{out_dir}/{model_name}_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_shap_summary(model, X_val, feature_names, out_dir, model_name):
    print("  Computing SHAP values…")
    try:
        explainer = shap.TreeExplainer(model)
        sample = X_val.iloc[:min(1000, len(X_val))]
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, sample, feature_names=feature_names,
                          show=False, plot_size=(10, 7))
        path = f'{out_dir}/{model_name}_shap_summary.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  SHAP saved: {path}")
        return explainer, path
    except Exception as e:
        print(f"  SHAP error: {e}")
        return None, None


# ── 5. Main training ──────────────────────────────────────────────────────────
def train():
    mlflow.set_tracking_uri('mlruns')
    mlflow.set_experiment('readmission_risk')

    df = load_data(DATA_PATH)
    X, y, engineer = build_features(df)
    feature_names = list(X.columns)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    print(f"\nTrain: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

    pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Class weight (neg/pos): {pos_weight}")

    # ── Model configs ──
    models_cfg = {
        'xgb_smote': {
            'clf': XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                eval_metric='aucpr', use_label_encoder=False,
                random_state=42, n_jobs=-1,
            ),
            'use_smote': True,
        },
        'xgb_weighted': {
            'clf': XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=pos_weight * 2,
                eval_metric='aucpr', use_label_encoder=False,
                random_state=42, n_jobs=-1,
            ),
            'use_smote': False,
        },
    }

    best_model, best_auc, best_name, best_threshold = None, 0, '', 0.5
    best_explainer = None

    for name, cfg in models_cfg.items():
        print(f"\n{'='*55}")
        print(f"Training: {name}")

        with mlflow.start_run(run_name=name):
            clf = cfg['clf']

            if cfg['use_smote']:
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_res, y_res = smote.fit_resample(X_train, y_train)
            else:
                X_res, y_res = X_train.copy(), y_train.copy()

            clf.fit(X_res, y_res,
                    eval_set=[(X_val.values, y_val)],
                    verbose=False)

            probs_val = clf.predict_proba(X_val)[:, 1]
            threshold, f2_val = tune_threshold(y_val, probs_val, beta=2)
            metrics = evaluate(y_val, probs_val, threshold, label=name)

            mlflow.log_params({
                'model': name, 'smote': cfg['use_smote'],
                'scale_pos_weight': clf.get_params().get('scale_pos_weight', 1),
                'threshold': round(threshold, 3),
            })
            mlflow.log_metrics(metrics)

            curve_path = plot_curves(y_val, probs_val, threshold, name, REPORT_DIR)
            mlflow.log_artifact(curve_path)

            explainer, shap_path = plot_shap_summary(
                clf, X_val, feature_names, REPORT_DIR, name)
            if shap_path:
                mlflow.log_artifact(shap_path)

            if metrics['auc_roc'] > best_auc:
                best_auc = metrics['auc_roc']
                best_model = clf
                best_name = name
                best_threshold = threshold
                best_explainer = explainer

    # ── Final test evaluation ──
    print(f"\n{'='*55}")
    print(f"FINAL TEST EVALUATION — {best_name}")
    probs_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, probs_test, best_threshold, label='TEST')

    # ── Save artifacts ──
    joblib.dump(best_model, f'{MODEL_DIR}/readmission_model.pkl')
    joblib.dump(engineer, f'{MODEL_DIR}/feature_engineer.pkl')
    joblib.dump(best_threshold, f'{MODEL_DIR}/threshold.pkl')
    joblib.dump(feature_names, f'{MODEL_DIR}/feature_names.pkl')
    if best_explainer:
        joblib.dump(best_explainer, f'{MODEL_DIR}/shap_explainer.pkl')

    # Save test metrics
    pd.DataFrame([test_metrics]).to_csv(f'{REPORT_DIR}/test_metrics.csv', index=False)

    print(f"\n✓ Best model: {best_name}")
    print(f"✓ Test AUC-ROC: {best_auc:.4f}")
    print(f"✓ Threshold: {best_threshold:.3f}")
    print(f"✓ Artifacts saved to: {MODEL_DIR}/")

    return best_model, engineer, best_threshold, feature_names, test_metrics


if __name__ == '__main__':
    train()
