"""
PulseAI — Hospital Readmission Risk Dashboard
Dark topbar + form strip using st.markdown for nav only.
All content below uses native Streamlit + targeted CSS overrides.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

st.set_page_config(
    page_title="PulseAI — Readmission Risk",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Hide ALL Streamlit chrome */
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }
button[kind="header"] { display: none !important; }

.block-container { padding: 0 2rem 2rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

.topbar {
    background: #0f172a; display: flex; align-items: center;
    justify-content: space-between; padding: 12px 0 16px;
    border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 16px;
}
.pulse-logo { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 500; color: #38bdf8; }
.pulse-logo b { color: #fff; font-weight: 300; }
.tbadges { display: flex; gap: 6px; }
.tbadge { font-size: 10px; padding: 3px 9px; border-radius: 20px; background: rgba(255,255,255,0.08); color: #94a3b8; }

.form-label { font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-bottom: 12px; }

.score-card { background: #0f172a; border-radius: 12px; padding: 28px 32px; }
.sc-eye { font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase; color: #475569; font-family: 'JetBrains Mono', monospace; margin-bottom: 8px; }
.sc-pct { font-size: 68px; font-weight: 300; line-height: 1; font-family: 'JetBrains Mono', monospace; }
.sc-pill { display: inline-block; font-size: 10px; padding: 4px 12px; border-radius: 20px; margin: 10px 0 14px; letter-spacing: 0.04em; font-weight: 500; }
.sc-rec { font-size: 12px; color: #64748b; line-height: 1.7; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.07); }

.mc { background: #fff; border: 0.5px solid #e2e8f0; border-radius: 8px; padding: 16px 18px; height: 90px; }
.mc-lbl { font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; }
.mc-val { font-size: 24px; font-weight: 500; color: #0f172a; font-family: 'JetBrains Mono', monospace; }
.mc-sub { font-size: 10px; color: #94a3b8; margin-top: 3px; }

.shap-card { background: #fff; border: 0.5px solid #e2e8f0; border-radius: 10px; padding: 20px; }
.sec-lbl { font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase; color: #94a3b8; font-family: 'JetBrains Mono', monospace; padding-bottom: 10px; border-bottom: 0.5px solid #f1f5f9; margin-bottom: 14px; }

.dr { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 0.5px solid #f8fafc; }
.dr-name { font-size: 11px; color: #334155; width: 150px; flex-shrink: 0; }
.dr-track { flex: 1; background: #f1f5f9; border-radius: 2px; height: 5px; }
.dr-pos { background: #0f172a; height: 5px; border-radius: 2px; }
.dr-neg { background: #f97316; height: 5px; border-radius: 2px; }
.dr-val { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #94a3b8; width: 46px; text-align: right; }

.roi-c { background: #ecfdf5; border: 0.5px solid #a7f3d0; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }
.roi-lbl { font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: #065f46; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }
.roi-val { font-size: 22px; font-weight: 500; color: #064e3b; font-family: 'JetBrains Mono', monospace; }
.roi-b { background: #eff6ff; border: 0.5px solid #bfdbfe; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }
.roi-lbl-b { font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: #1e40af; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }
.roi-val-b { font-size: 22px; font-weight: 500; color: #1e3a8a; font-family: 'JetBrains Mono', monospace; }

.landing-hero { background: #0f172a; border-radius: 14px; padding: 44px 40px; margin-bottom: 20px; }
.lh-tag { font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase; color: #38bdf8; font-family: 'JetBrains Mono', monospace; margin-bottom: 10px; }
.lh-h1 { font-size: 38px; font-weight: 300; color: #fff; margin-bottom: 8px; }
.lh-h1 b { font-weight: 500; color: #38bdf8; }
.lh-sub { font-size: 14px; color: #64748b; line-height: 1.7; margin-bottom: 28px; }
.perf-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
.perf-cell { background: rgba(255,255,255,0.04); border: 0.5px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 16px; text-align: center; }
.perf-val { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 500; color: #38bdf8; }
.perf-lbl { font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-top: 4px; }

.how-card { background: #fff; border: 0.5px solid #e2e8f0; border-radius: 10px; padding: 20px; height: 140px; }
.how-num { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; margin-bottom: 8px; }
.how-title { font-size: 13px; font-weight: 500; color: #0f172a; margin-bottom: 6px; }
.how-body { font-size: 12px; color: #64748b; line-height: 1.6; }

div[data-testid="stSlider"] > div > div > div { background: #0f172a !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(__file__))
    md = os.path.join(base, 'models')
    try:
        return (
            joblib.load(f'{md}/readmission_model.pkl'),
            joblib.load(f'{md}/feature_engineer.pkl'),
            joblib.load(f'{md}/shap_explainer.pkl'),
            joblib.load(f'{md}/feature_names.pkl'),
            joblib.load(f'{md}/threshold.pkl'),
            joblib.load(f'{md}/test_metrics.pkl'),
        )
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, None, 0.35, {}

model, engineer, explainer, feature_names, threshold, test_metrics = load_artifacts()

# ── Top nav ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="pulse-logo">Pulse<b>AI</b></div>
  <div class="tbadges">
    <span class="tbadge">Readmission Intelligence</span>
    <span class="tbadge">XGBoost · SHAP</span>
    <span class="tbadge">CMS HRRP</span>
  </div>
</div>
<div class="form-label">Patient Assessment</div>
""", unsafe_allow_html=True)

# ── Patient form ──────────────────────────────────────────────────────────────
r1 = st.columns([1,1,1,1,1,1,1,1,1])
with r1[0]: age = st.selectbox("Age", ['[40-50)','[50-60)','[60-70)','[70-80)','[80-90)'], index=2)
with r1[1]: admission_type = st.selectbox("Admission", [1,2,3], format_func=lambda x:{1:"Emergency",2:"Urgent",3:"Elective"}[x])
with r1[2]: time_in_hospital = st.slider("LOS days", 1, 14, 5)
with r1[3]: num_medications = st.slider("Medications", 1, 60, 16)
with r1[4]: number_inpatient = st.slider("Prior inpatient", 0, 10, 0)
with r1[5]: number_emergency = st.slider("Prior ER", 0, 10, 0)
with r1[6]: a1c = st.selectbox("HbA1c", ["None",">7",">8","Norm"])
with r1[7]: diag_1 = st.selectbox("ICD-9", ["250.00","250.40","414.01","428.0","486","585.3","401.9"])
with r1[8]:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Assessment", use_container_width=True, type="primary")

r2 = st.columns([1,1,1,1])
with r2[0]: number_diagnoses = st.slider("Diagnoses", 1, 9, 7)
with r2[1]: num_lab_procedures = st.slider("Lab procedures", 0, 100, 43)
with r2[2]:
    insulin = st.selectbox("Insulin", ["No","Steady","Up","Down"])
    change = st.selectbox("Med change", ["No","Ch"])
with r2[3]:
    diab_med = st.selectbox("Diabetes med", ["Yes","No"])
    gender = st.radio("Gender", ["Female","Male"], horizontal=True)

st.divider()

# ── Results / Landing ─────────────────────────────────────────────────────────
if run_btn and model is not None:
    row = {
        'age': age, 'gender': gender, 'race': 'Caucasian',
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': 1, 'num_medications': num_medications,
        'number_outpatient': 0, 'number_emergency': number_emergency,
        'number_inpatient': number_inpatient, 'number_diagnoses': number_diagnoses,
        'diag_1': diag_1, 'diag_2': '?', 'diag_3': '?',
        'admission_type_id': admission_type,
        'discharge_disposition_id': 1, 'admission_source_id': 4,
        'A1Cresult': a1c, 'change': change, 'diabetesMed': diab_med,
        'insulin': insulin, 'metformin':'No','repaglinide':'No',
        'nateglinide':'No','glimepiride':'No','glipizide':'No','glyburide':'No',
        'weight':'?','payer_code':'MC','medical_specialty':'InternalMedicine',
    }
    X = engineer.transform(pd.DataFrame([row]))
    prob = float(model.predict_proba(X.values)[0][1])

    if prob >= 0.60:
        tier, pct_color = "High Risk", "#f97316"
        pill_style = "background:rgba(249,115,22,0.13);color:#c2410c"
        rec = "Assign care manager · 48h post-discharge call · Medication reconciliation · Social work consult"
    elif prob >= 0.38:
        tier, pct_color = "Moderate Risk", "#eab308"
        pill_style = "background:rgba(234,179,8,0.13);color:#854d0e"
        rec = "7-day clinic follow-up · Pharmacist review · Patient education · Community health worker"
    else:
        tier, pct_color = "Low Risk", "#22c55e"
        pill_style = "background:rgba(34,197,94,0.12);color:#14532d"
        rec = "Standard discharge protocol · 30-day outpatient follow-up"

    flag = prob >= float(threshold)
    avoided = int(prob * 15200 * 0.40)
    ann_avoided = int(5000 * 0.15 * 0.72 * 0.40)

    # Row 1: score + metrics
    col_score, col_m = st.columns([2, 3])
    with col_score:
        st.markdown(f"""
        <div class="score-card">
          <div class="sc-eye">30-day readmission probability</div>
          <div class="sc-pct" style="color:{pct_color}">{prob*100:.0f}%</div>
          <span class="sc-pill" style="{pill_style}">{tier}</span>
          <div class="sc-rec">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m:
        ma, mb = st.columns(2)
        mc_, md = st.columns(2)
        fc = "#e05050" if flag else "#22c55e"
        ft = "FLAG" if flag else "PASS"
        with ma:
            st.markdown(f'<div class="mc"><div class="mc-lbl">Exact score</div><div class="mc-val">{prob*100:.1f}%</div><div class="mc-sub">Model output</div></div>', unsafe_allow_html=True)
        with mb:
            st.markdown(f'<div class="mc"><div class="mc-lbl">Decision</div><div class="mc-val" style="color:{fc}">{ft}</div><div class="mc-sub">At {float(threshold)*100:.0f}% threshold</div></div>', unsafe_allow_html=True)
        with mc_:
            st.markdown(f'<div class="mc"><div class="mc-lbl">Avoided cost</div><div class="mc-val">${avoided:,}</div><div class="mc-sub">If intervention succeeds</div></div>', unsafe_allow_html=True)
        with md:
            st.markdown(f'<div class="mc"><div class="mc-lbl">AUC-ROC</div><div class="mc-val">{test_metrics.get("auc_roc",0):.3f}</div><div class="mc-sub">Held-out test set</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: SHAP bars + ROI column
    col_shap, col_roi = st.columns([3, 1])
    with col_shap:
        st.markdown('<div class="shap-card"><div class="sec-lbl">SHAP feature contributions — why this patient is flagged</div>', unsafe_allow_html=True)
        if explainer:
            try:
                sv = explainer.shap_values(X.values)[0]
                top_idx = np.argsort(np.abs(sv))[::-1][:10]
                max_abs = max(abs(sv[i]) for i in top_idx) + 1e-9
                lc, rc = st.columns(2)
                for col, idxs in [(lc, top_idx[:5]), (rc, top_idx[5:])]:
                    with col:
                        rows_html = ""
                        for i in idxs:
                            name = feature_names[i].replace('_',' ')
                            val = sv[i]
                            pct = int(abs(val)/max_abs*100)
                            fill = "dr-pos" if val > 0 else "dr-neg"
                            sign = "+" if val > 0 else ""
                            rows_html += f'<div class="dr"><div class="dr-name">{name}</div><div class="dr-track"><div class="{fill}" style="width:{pct}%"></div></div><div class="dr-val">{sign}{val:.3f}</div></div>'
                        st.markdown(rows_html, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"SHAP error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_roi:
        st.markdown(f"""
        <div class="roi-c"><div class="roi-lbl">Annual savings</div><div class="roi-val">${ann_avoided*15200/1e6:.1f}M</div></div>
        <div class="roi-c"><div class="roi-lbl">Readmissions avoided</div><div class="roi-val">{ann_avoided}/yr</div></div>
        <div class="roi-b"><div class="roi-lbl-b">Sensitivity</div><div class="roi-val-b">{test_metrics.get('sensitivity',0):.3f}</div></div>
        <div class="roi-b"><div class="roi-lbl-b">F2 Score</div><div class="roi-val-b">{test_metrics.get('f2',0):.3f}</div></div>
        """, unsafe_allow_html=True)

    # Row 3: waterfall
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="shap-card"><div class="sec-lbl">Cumulative SHAP waterfall — patient-level breakdown</div>', unsafe_allow_html=True)
    if explainer:
        try:
            sv = explainer.shap_values(X.values)[0]
            shap_exp = shap.Explanation(
                values=sv, base_values=float(explainer.expected_value),
                data=X.values[0], feature_names=feature_names)
            fig, _ = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('#ffffff')
            shap.waterfall_plot(shap_exp, show=False, max_display=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close('all')
        except Exception as e:
            st.caption(f"Waterfall unavailable: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Landing
    m = test_metrics
    st.markdown(f"""
    <div class="landing-hero">
      <div class="lh-tag">Hospital Readmission Intelligence</div>
      <div class="lh-h1">Predict. Prevent. <b>Save Lives.</b></div>
      <div class="lh-sub">PulseAI scores 30-day readmission risk from EHR features using XGBoost + SHAP — giving care teams the <em>why</em> behind every flag, not just a number.</div>
      <div class="perf-row">
        <div class="perf-cell"><div class="perf-val">{m.get('auc_roc',0):.3f}</div><div class="perf-lbl">AUC-ROC</div></div>
        <div class="perf-cell"><div class="perf-val">{m.get('auc_pr',0):.3f}</div><div class="perf-lbl">AUC-PR</div></div>
        <div class="perf-cell"><div class="perf-val">{m.get('f2',0):.3f}</div><div class="perf-lbl">F2 Score</div></div>
        <div class="perf-cell"><div class="perf-val">{m.get('sensitivity',0):.3f}</div><div class="perf-lbl">Sensitivity</div></div>
        <div class="perf-cell"><div class="perf-val">$15.2K</div><div class="perf-lbl">Cost / readmit</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    for col, num, title, body in [
        (h1, "01", "Input patient data", "Fill the form above in under 60 seconds. Age, LOS, medications, prior visits, HbA1c, ICD-9. No manual feature mapping needed."),
        (h2, "02", "Risk score generated", "XGBoost scores 30-day readmission probability. Threshold tuned for F2-score — missing a high-risk patient costs more than a false alarm."),
        (h3, "03", "SHAP explains the why", "Top 10 feature contributions as ranked bars + waterfall. Clinicians see exactly what drove the flag, not a black-box number."),
    ]:
        with col:
            st.markdown(f'<div class="how-card"><div class="how-num">{num}</div><div class="how-title">{title}</div><div class="how-body">{body}</div></div>', unsafe_allow_html=True)

    base = os.path.dirname(os.path.dirname(__file__))
    img = os.path.join(base, 'reports', 'eval_curves.png')
    if os.path.exists(img):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="shap-card"><div class="sec-lbl">Evaluation curves — held-out test set</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
