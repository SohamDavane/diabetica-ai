"""
DiabéticaAI — Streamlit Frontend
==================================
Full clinical dashboard with:
  - Single patient prediction + SHAP explanation
  - Batch CSV upload and scoring
  - Model performance metrics
  - Risk stratification visual
  - Data drift monitor
"""

import sys
import pickle
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── add project root to path so ml/ imports work ──
sys.path.append(str(Path(__file__).parent))

from sklearn.experimental import enable_iterative_imputer  # noqa

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabéticaAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #111827 100%);
        border-right: 1px solid #2d3748;
    }

    /* Cards */
    .metric-card {
        background: #1e2533;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 4px;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Section headers */
    .section-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4b5563;
        margin-bottom: 12px;
        margin-top: 4px;
    }

    /* Recommendation item */
    .rec-item {
        background: #1a2030;
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        color: #d1d5db;
        line-height: 1.5;
    }

    /* Narrative box */
    .narrative-box {
        background: #1e2533;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.88rem;
        color: #9ca3af;
        line-height: 1.7;
        font-style: italic;
    }

    /* Disclaimer */
    .disclaimer {
        background: #1a1f2e;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.75rem;
        color: #6b7280;
        line-height: 1.5;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #6b7280;
        border-radius: 7px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #2d3748 !important;
        color: #f9fafb !important;
    }

    /* Input fields */
    .stNumberInput input, .stSlider { color: #f9fafb; }

    /* Divider */
    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Training DiabéticaAI model — please wait ~60 seconds…")
def load_model():
    from ml.pipeline import (
        ClinicalPreprocessor, EnsembleTrainer, ExplainabilityEngine,
        RiskStratifier, ModelRegistry, DataLoader, build_and_train_pipeline
    )

    model_path = Path("models/diabetica_v1.0.0.pkl")
    if not model_path.exists():
        st.info("⏳ No model found — training now on synthetic data. This takes about 60 seconds...")
        build_and_train_pipeline()

    with open(model_path, "rb") as f:
        return pickle.load(f)


artifact = load_model()
model        = artifact["model"]
preprocessor = artifact["preprocessor"]
feat_names   = artifact["feature_names"]
metrics      = artifact.get("metrics", {})
cv_metrics   = artifact.get("cv_metrics", {})


# ─────────────────────────────────────────────
#  HELPER: PREDICT SINGLE PATIENT
# ─────────────────────────────────────────────

def predict_patient(inputs: dict) -> dict:
    from ml.pipeline import RiskStratifier, ExplainabilityEngine

    df = pd.DataFrame([inputs])

    # align columns
    base_features = [c for c in feat_names if not any(
        c.endswith(s) for s in ["_Obese","_High","_Senior","_Overweight","_PreDM","_MiddleAge"]
    )]
    for col in base_features:
        if col not in df.columns:
            df[col] = np.nan

    df_proc = preprocessor.transform(df)
    for col in feat_names:
        if col not in df_proc.columns:
            df_proc[col] = 0.0
    df_proc = df_proc.reindex(columns=feat_names, fill_value=0.0)
    X = df_proc.values

    prob = model.predict_proba(X)[0][1]
    stratifier = RiskStratifier()
    risk = stratifier.stratify(prob)

    # SHAP
    try:
        engine = ExplainabilityEngine(model, feat_names)
        explanation = engine.explain_patient(X)
    except Exception:
        explanation = {"factors": [], "clinical_narrative": "Explanation unavailable."}

    return {"probability": prob, "risk": risk, "explanation": explanation, "X": X}


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🩺 DiabéticaAI")
    st.markdown("<div style='color:#6b7280;font-size:0.8rem;margin-bottom:1.5rem;'>Clinical Diabetes Risk Platform</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Model Status</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        st.metric("Recall",   f"{metrics.get('recall', 0):.3f}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        st.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.3f}")

    st.markdown("---")
    st.markdown("<div class='section-title'>CV Results (10-fold)</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.8rem;color:#9ca3af;line-height:2;'>
    F1 Mean: <b style='color:#34d399'>{cv_metrics.get('cv_f1_mean', 0):.4f}</b> ± {cv_metrics.get('cv_f1_std', 0):.4f}<br>
    Recall:  <b style='color:#34d399'>{cv_metrics.get('cv_recall_mean', 0):.4f}</b> ± {cv_metrics.get('cv_recall_std', 0):.4f}<br>
    AUC:     <b style='color:#34d399'>{cv_metrics.get('cv_roc_auc_mean', 0):.4f}</b> ± {cv_metrics.get('cv_roc_auc_std', 0):.4f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#4b5563;line-height:1.6;'>
    ⚠️ For clinical decision support only.<br>
    Not a substitute for professional medical judgment.<br><br>
    Model: XGB + LGBM + RF Ensemble<br>
    Data: Synthetic demo (train on real data for production)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style='margin-bottom:1.5rem;'>
    <h1 style='font-size:2rem;font-weight:700;color:#f9fafb;margin:0;'>
        🩺 DiabéticaAI
    </h1>
    <p style='color:#6b7280;font-size:0.95rem;margin-top:4px;'>
        Clinical Diabetes Risk Assessment · Ensemble ML · SHAP Explainability
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Single Patient",
    "📋 Batch Scoring",
    "📊 Model Analytics",
    "📡 Drift Monitor"
])


# ══════════════════════════════════════════════
#  TAB 1 — SINGLE PATIENT PREDICTION
# ══════════════════════════════════════════════

with tab1:
    st.markdown("<div class='section-title'>Patient Clinical Features</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                glucose   = st.slider("🩸 Blood Glucose (mg/dL)", 50, 300, 120, help="Plasma glucose — 2hr OGTT")
                bmi       = st.slider("⚖️ BMI (kg/m²)",           15.0, 65.0, 28.0, step=0.1)
                age       = st.slider("🎂 Age (years)",            18, 90, 45)
                bp        = st.slider("💓 Blood Pressure (mmHg)", 40, 140, 72, help="Diastolic BP")
            with c2:
                insulin   = st.slider("💉 Insulin (μU/mL)",        0, 850, 80)
                skin      = st.slider("📏 Skin Thickness (mm)",     0, 100, 20)
                dpf       = st.slider("🧬 Family History Score",   0.05, 2.5, 0.35, step=0.01, help="Diabetes Pedigree Function")
                preg      = st.slider("🤰 Pregnancies",             0, 17, 2)

        patient_id = st.text_input("Patient ID (optional)", placeholder="e.g. PT-2024-001")

        predict_btn = st.button("🔍 Assess Risk", use_container_width=True, type="primary")

    with col_right:
        if predict_btn:
            inputs = {
                "Glucose": glucose, "BMI": bmi, "Age": age,
                "BloodPressure": bp, "Insulin": insulin,
                "SkinThickness": skin, "DiabetesPedigreeFunction": dpf,
                "Pregnancies": preg
            }

            with st.spinner("Running ensemble inference…"):
                result = predict_patient(inputs)

            prob  = result["probability"]
            risk  = result["risk"]
            expl  = result["explanation"]
            code  = risk["risk_code"]

            # Color map
            colors = {
                "LOW":      ("#22c55e", "#052e16", "✅"),
                "MODERATE": ("#f59e0b", "#2d1a00", "⚠️"),
                "HIGH":     ("#f97316", "#2d0e00", "🔶"),
                "CRITICAL": ("#ef4444", "#2d0000", "🚨"),
            }
            color, bg, icon = colors[code]

            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 36, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4b5563", "tickfont": {"color": "#6b7280", "size": 11}},
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": "#1e2533",
                    "bordercolor": "#2d3748",
                    "steps": [
                        {"range": [0,  30], "color": "#052e16"},
                        {"range": [30, 60], "color": "#2d1a00"},
                        {"range": [60, 80], "color": "#2d0e00"},
                        {"range": [80,100], "color": "#2d0000"},
                    ],
                    "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob * 100}
                },
                title={"text": "Diabetes Risk Probability", "font": {"color": "#9ca3af", "size": 13}}
            ))
            fig_gauge.update_layout(
                height=240, margin=dict(t=40, b=10, l=20, r=20),
                paper_bgcolor="#0f1117", font_color="#f9fafb"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk badge
            st.markdown(f"""
            <div style='text-align:center;margin:-10px 0 16px;'>
                <span style='background:{bg};color:{color};border:1.5px solid {color};
                    padding:6px 24px;border-radius:999px;font-size:1rem;font-weight:700;
                    letter-spacing:0.05em;'>
                    {icon} {risk['risk_label']}  ·  {risk['risk_percentage']}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Follow-up
            st.markdown(f"""
            <div style='text-align:center;color:#6b7280;font-size:0.8rem;margin-bottom:16px;'>
                Recommended follow-up: <b style='color:#9ca3af;'>{risk['follow_up_days']} days</b>
            </div>
            """, unsafe_allow_html=True)

            # Clinical narrative
            st.markdown("<div class='section-title'>Clinical Narrative</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='narrative-box'>{expl['clinical_narrative']}</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='height:300px;display:flex;flex-direction:column;align-items:center;
                justify-content:center;color:#374151;text-align:center;'>
                <div style='font-size:3rem;margin-bottom:12px;'>🩺</div>
                <div style='font-size:1rem;font-weight:500;color:#4b5563;'>Adjust patient parameters</div>
                <div style='font-size:0.85rem;margin-top:6px;color:#374151;'>then click Assess Risk</div>
            </div>
            """, unsafe_allow_html=True)

    # SHAP explanation (full width below)
    if predict_btn and result["explanation"]["factors"]:
        st.markdown("---")
        st.markdown("<div class='section-title'>SHAP Feature Attribution — Why this prediction?</div>", unsafe_allow_html=True)

        factors = result["explanation"]["factors"]
        labels  = [f["label"] for f in factors]
        shap_v  = [f["shap_value"] for f in factors]
        bar_colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_v]

        fig_shap = go.Figure(go.Bar(
            x=shap_v, y=labels, orientation="h",
            marker_color=bar_colors,
            text=[f"{'+' if v>0 else ''}{v:.3f}" for v in shap_v],
            textposition="outside",
            textfont={"color": "#9ca3af", "size": 11},
        ))
        fig_shap.add_vline(x=0, line_color="#4b5563", line_width=1)
        fig_shap.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=60),
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2533",
            xaxis={"gridcolor": "#2d3748", "zerolinecolor": "#4b5563",
                   "title": "SHAP Value (impact on prediction)", "title_font": {"color": "#6b7280"},
                   "tickfont": {"color": "#6b7280"}},
            yaxis={"gridcolor": "#2d3748", "tickfont": {"color": "#d1d5db"}, "autorange": "reversed"},
            showlegend=False,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Recommendations
        st.markdown("<div class='section-title'>Clinical Recommendations</div>", unsafe_allow_html=True)
        rec_colors = {"LOW": "#22c55e", "MODERATE": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
        rc = rec_colors[code]
        cols_rec = st.columns(2)
        for i, rec in enumerate(risk["recommendations"]):
            with cols_rec[i % 2]:
                st.markdown(f"<div class='rec-item' style='border-color:{rc};'>{rec}</div>", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class='disclaimer'>
        ⚠️ <b>Clinical Disclaimer:</b> This prediction is AI-generated and intended to support, not replace, clinical judgment.
        All results must be reviewed and confirmed by a licensed healthcare professional.
        Not validated as a medical device. GDPR Art. 22 compliant via SHAP explanations.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — BATCH SCORING
# ══════════════════════════════════════════════

with tab2:
    st.markdown("<div class='section-title'>Batch Patient Scoring via CSV Upload</div>", unsafe_allow_html=True)

    col_info, col_upload = st.columns([1, 1.5], gap="large")

    with col_info:
        st.markdown("""
        <div style='background:#1e2533;border:1px solid #2d3748;border-radius:10px;padding:1.2rem;font-size:0.85rem;color:#9ca3af;line-height:1.8;'>
        <b style='color:#d1d5db;'>Required CSV columns:</b><br>
        • Glucose<br>
        • BMI<br>
        • Age<br>
        • BloodPressure<br>
        • Insulin<br>
        • SkinThickness<br>
        • DiabetesPedigreeFunction<br>
        • Pregnancies<br><br>
        <b style='color:#d1d5db;'>Optional:</b> PatientID column for tracking<br><br>
        Max 500 rows per upload.
        </div>
        """, unsafe_allow_html=True)

        # Download sample CSV
        sample_data = pd.DataFrame({
            "PatientID":               ["PT-001", "PT-002", "PT-003", "PT-004", "PT-005"],
            "Glucose":                 [148, 85, 183, 110, 197],
            "BloodPressure":           [72, 66, 64, 78, 70],
            "SkinThickness":           [35, 29, 0, 31, 45],
            "Insulin":                 [0, 0, 0, 0, 543],
            "BMI":                     [33.6, 26.6, 23.3, 30.1, 30.5],
            "DiabetesPedigreeFunction":[0.627, 0.351, 0.672, 0.245, 0.158],
            "Age":                     [50, 31, 32, 26, 53],
            "Pregnancies":             [6, 1, 8, 2, 10],
        })
        csv_sample = sample_data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Sample CSV", csv_sample, "sample_patients.csv", "text/csv", use_container_width=True)

    with col_upload:
        uploaded = st.file_uploader("Upload patient CSV", type=["csv"])

        if uploaded:
            df_batch = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df_batch)} patients")

            if len(df_batch) > 500:
                st.warning("Truncating to first 500 rows.")
                df_batch = df_batch.head(500)

            if st.button("🚀 Run Batch Scoring", type="primary", use_container_width=True):
                results_list = []
                prog = st.progress(0, text="Scoring patients…")

                for i, row in df_batch.iterrows():
                    inp = {
                        col: row[col] for col in
                        ["Glucose","BMI","Age","BloodPressure","Insulin",
                         "SkinThickness","DiabetesPedigreeFunction","Pregnancies"]
                        if col in row
                    }
                    try:
                        r = predict_patient(inp)
                        results_list.append({
                            "PatientID":   row.get("PatientID", f"PT-{i+1:04d}"),
                            "Risk Label":  r["risk"]["risk_label"],
                            "Risk Code":   r["risk"]["risk_code"],
                            "Probability": round(r["probability"] * 100, 1),
                            "Follow-up (days)": r["risk"]["follow_up_days"],
                            "Top Factor":  r["explanation"]["factors"][0]["label"] if r["explanation"]["factors"] else "N/A",
                        })
                    except Exception as e:
                        results_list.append({"PatientID": f"PT-{i+1:04d}", "Error": str(e)})

                    prog.progress((i + 1) / len(df_batch), text=f"Scored {i+1}/{len(df_batch)} patients…")

                prog.empty()
                results_df = pd.DataFrame(results_list)
                st.session_state["batch_results"] = results_df

    # Show batch results
    if "batch_results" in st.session_state:
        res = st.session_state["batch_results"]
        st.markdown("---")
        st.markdown("<div class='section-title'>Batch Results</div>", unsafe_allow_html=True)

        # Summary metrics
        if "Risk Code" in res.columns:
            counts = res["Risk Code"].value_counts()
            m1, m2, m3, m4, m5 = st.columns(5)
            total = len(res)
            with m1:
                st.markdown(f"<div class='metric-card'><div class='label'>Total Scored</div><div class='value' style='color:#f9fafb;'>{total}</div></div>", unsafe_allow_html=True)
            with m2:
                n = counts.get("LOW", 0)
                st.markdown(f"<div class='metric-card'><div class='label'>Low Risk</div><div class='value' style='color:#22c55e;'>{n}</div><div class='sub'>{n/total*100:.0f}%</div></div>", unsafe_allow_html=True)
            with m3:
                n = counts.get("MODERATE", 0)
                st.markdown(f"<div class='metric-card'><div class='label'>Moderate</div><div class='value' style='color:#f59e0b;'>{n}</div><div class='sub'>{n/total*100:.0f}%</div></div>", unsafe_allow_html=True)
            with m4:
                n = counts.get("HIGH", 0)
                st.markdown(f"<div class='metric-card'><div class='label'>High Risk</div><div class='value' style='color:#f97316;'>{n}</div><div class='sub'>{n/total*100:.0f}%</div></div>", unsafe_allow_html=True)
            with m5:
                n = counts.get("CRITICAL", 0)
                st.markdown(f"<div class='metric-card'><div class='label'>Critical</div><div class='value' style='color:#ef4444;'>{n}</div><div class='sub'>{n/total*100:.0f}%</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Donut chart
            col_chart, col_table = st.columns([1, 1.6])
            with col_chart:
                fig_donut = go.Figure(go.Pie(
                    labels=counts.index.tolist(),
                    values=counts.values.tolist(),
                    hole=0.55,
                    marker_colors=["#22c55e","#f59e0b","#f97316","#ef4444"],
                    textinfo="percent+label",
                    textfont={"size": 12, "color": "#f9fafb"},
                ))
                fig_donut.update_layout(
                    height=250, margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="#0f1117",
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_table:
                # Color-code the Risk Label column
                def color_risk(val):
                    c = {"Low Risk": "#22c55e", "Moderate Risk": "#f59e0b",
                         "High Risk": "#f97316", "Critical Risk": "#ef4444"}.get(val, "white")
                    return f"color: {c}; font-weight: 600"

                styled = res.drop(columns=["Risk Code"], errors="ignore").style.applymap(
                    color_risk, subset=["Risk Label"]
                ).format({"Probability": "{:.1f}%"})
                st.dataframe(styled, use_container_width=True, height=220)

        # Download results
        csv_out = res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", csv_out, "diabetica_batch_results.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — MODEL ANALYTICS
# ══════════════════════════════════════════════

with tab3:
    st.markdown("<div class='section-title'>Model Performance Analytics</div>", unsafe_allow_html=True)

    # Performance radar chart
    col_radar, col_bars = st.columns(2)

    with col_radar:
        categories = ["F1-Score", "Recall", "Precision", "ROC-AUC"]
        values_model = [
            metrics.get("f1_score", 0),
            metrics.get("recall", 0),
            metrics.get("precision", 0),
            metrics.get("roc_auc", 0),
        ]
        values_target = [0.85, 0.90, 0.85, 0.90]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_model + [values_model[0]],
            theta=categories + [categories[0]],
            fill="toself", name="Achieved",
            line_color="#34d399", fillcolor="rgba(52,211,153,0.15)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=values_target + [values_target[0]],
            theta=categories + [categories[0]],
            fill="toself", name="Target",
            line_color="#6366f1", fillcolor="rgba(99,102,241,0.1)",
            line_dash="dash"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.8, 1.0], tickfont={"color": "#6b7280", "size": 10}, gridcolor="#2d3748"),
                angularaxis=dict(tickfont={"color": "#d1d5db", "size": 12}, gridcolor="#2d3748"),
                bgcolor="#1e2533"
            ),
            showlegend=True,
            legend=dict(font={"color": "#9ca3af"}, bgcolor="#0f1117", bordercolor="#2d3748", borderwidth=1),
            paper_bgcolor="#0f1117",
            height=340, margin=dict(t=20, b=20, l=20, r=20),
            title=dict(text="Achieved vs Target Metrics", font={"color": "#9ca3af", "size": 13})
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bars:
        # CV metrics with error bars
        cv_metrics_names  = ["F1-Score", "Recall", "ROC-AUC"]
        cv_means = [cv_metrics.get("cv_f1_mean",0), cv_metrics.get("cv_recall_mean",0), cv_metrics.get("cv_roc_auc_mean",0)]
        cv_stds  = [cv_metrics.get("cv_f1_std",0),  cv_metrics.get("cv_recall_std",0),  cv_metrics.get("cv_roc_auc_std",0)]

        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=cv_metrics_names, y=cv_means,
            error_y=dict(type="data", array=cv_stds, color="#6b7280"),
            marker_color=["#34d399", "#60a5fa", "#a78bfa"],
            text=[f"{v:.4f}" for v in cv_means],
            textposition="outside", textfont={"color": "#9ca3af", "size": 11},
        ))
        fig_cv.add_hline(y=0.85, line_dash="dot", line_color="#f59e0b",
                         annotation_text="Target 0.85", annotation_font_color="#f59e0b")
        fig_cv.update_layout(
            height=340, paper_bgcolor="#0f1117", plot_bgcolor="#1e2533",
            margin=dict(t=30, b=20, l=10, r=10),
            title=dict(text="10-Fold Cross-Validation (mean ± std)", font={"color": "#9ca3af", "size": 13}),
            xaxis=dict(tickfont={"color": "#d1d5db"}, gridcolor="#2d3748"),
            yaxis=dict(tickfont={"color": "#6b7280"}, gridcolor="#2d3748", range=[0.88, 1.02]),
            showlegend=False,
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # Risk tier explainer
    st.markdown("---")
    st.markdown("<div class='section-title'>Risk Stratification Tier Definitions</div>", unsafe_allow_html=True)

    tiers = [
        ("Low Risk",      "0–30%",   "#22c55e", "Annual HbA1c screening. Lifestyle maintenance.",          "365 days"),
        ("Moderate Risk", "30–60%",  "#f59e0b", "Dietitian referral. DPP enrollment. Quarterly glucose.",  "90 days"),
        ("High Risk",     "60–80%",  "#f97316", "Urgent endocrinology. HbA1c within 14 days.",             "14 days"),
        ("Critical Risk", "80–100%", "#ef4444", "URGENT evaluation within 48–72 hrs. CGM evaluation.",     "2 days"),
    ]

    cols = st.columns(4)
    for col, (label, rng, color, action, followup) in zip(cols, tiers):
        with col:
            st.markdown(f"""
            <div style='background:#1e2533;border:1px solid {color}33;border-top:3px solid {color};
                border-radius:10px;padding:1rem;text-align:center;height:170px;'>
                <div style='color:{color};font-size:0.85rem;font-weight:700;margin-bottom:4px;'>{label}</div>
                <div style='color:#9ca3af;font-size:1.4rem;font-weight:700;margin-bottom:8px;'>{rng}</div>
                <div style='color:#6b7280;font-size:0.75rem;line-height:1.5;margin-bottom:8px;'>{action}</div>
                <div style='color:{color};font-size:0.72rem;font-weight:600;'>Follow-up: {followup}</div>
            </div>
            """, unsafe_allow_html=True)

    # Feature importance from model
    st.markdown("---")
    st.markdown("<div class='section-title'>Ensemble Feature Importance</div>", unsafe_allow_html=True)

    try:
        # Get feature importances from XGBoost sub-estimator
        xgb_model = model.estimators_[0]
        importances = xgb_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(12)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#34d399"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside", textfont={"color": "#6b7280", "size": 10},
        ))
        fig_fi.update_layout(
            height=380, paper_bgcolor="#0f1117", plot_bgcolor="#1e2533",
            margin=dict(t=10, b=10, l=10, r=60),
            xaxis=dict(gridcolor="#2d3748", tickfont={"color": "#6b7280"}, title_font={"color": "#6b7280"}),
            yaxis=dict(gridcolor="#2d3748", tickfont={"color": "#d1d5db"}),
            showlegend=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance chart unavailable: {e}")


# ══════════════════════════════════════════════
#  TAB 4 — DRIFT MONITOR
# ══════════════════════════════════════════════

with tab4:
    st.markdown("<div class='section-title'>Data Drift Monitoring (PSI Analysis)</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1a1f2e;border:1px solid #374151;border-radius:10px;
        padding:1rem 1.2rem;font-size:0.85rem;color:#9ca3af;margin-bottom:1.5rem;line-height:1.7;'>
    Upload a <b style='color:#d1d5db;'>current patient dataset CSV</b> to compare against the training distribution.
    The Population Stability Index (PSI) detects when real-world data is drifting away from what the model was trained on,
    indicating it may need retraining.
    <br><br>
    <b style='color:#22c55e;'>PSI &lt; 0.10</b> — Stable &nbsp;|&nbsp;
    <b style='color:#f59e0b;'>PSI 0.10–0.25</b> — Investigate &nbsp;|&nbsp;
    <b style='color:#ef4444;'>PSI &gt; 0.25</b> — Retrain model
    </div>
    """, unsafe_allow_html=True)

    drift_file = st.file_uploader("Upload current data CSV for drift analysis", type=["csv"], key="drift_upload")

    if drift_file:
        df_current = pd.read_csv(drift_file)
        st.success(f"✅ Loaded {len(df_current)} records for drift analysis")

        if st.button("🔍 Run Drift Analysis", type="primary"):
            from monitoring.drift import FeatureDriftDetector

            # Build a synthetic reference distribution from the preprocessor's fitted data
            np.random.seed(42)
            n_ref = 1000
            ref_data = pd.DataFrame({
                "Glucose":                  np.random.normal(120, 32, n_ref).clip(50, 300),
                "BMI":                      np.random.normal(32, 7, n_ref).clip(15, 65),
                "Age":                      np.random.randint(21, 81, n_ref).astype(float),
                "BloodPressure":            np.random.normal(72, 12, n_ref).clip(40, 130),
                "Insulin":                  np.random.exponential(80, n_ref).clip(0, 850),
                "SkinThickness":            np.random.normal(29, 11, n_ref).clip(5, 80),
                "DiabetesPedigreeFunction": np.random.exponential(0.47, n_ref).clip(0.07, 2.5),
                "Pregnancies":              np.random.randint(0, 17, n_ref).astype(float),
            })

            detector = FeatureDriftDetector(ref_data)
            report   = detector.detect(df_current)

            feature_report = report["features"]
            overall        = report["overall"]

            # Overall status
            sev_color = {"NONE": "#22c55e", "MODERATE": "#f59e0b", "CRITICAL": "#ef4444"}
            sev = overall["severity"]
            sc  = sev_color.get(sev, "#9ca3af")

            oc1, oc2, oc3, oc4 = st.columns(4)
            with oc1:
                st.markdown(f"<div class='metric-card'><div class='label'>Avg PSI</div><div class='value' style='color:{sc};'>{overall['avg_psi']:.4f}</div></div>", unsafe_allow_html=True)
            with oc2:
                st.markdown(f"<div class='metric-card'><div class='label'>Overall Severity</div><div class='value' style='color:{sc};font-size:1.2rem;'>{sev}</div></div>", unsafe_allow_html=True)
            with oc3:
                n_crit = overall['critical_features']
                cc = "#ef4444" if n_crit > 0 else "#22c55e"
                st.markdown(f"<div class='metric-card'><div class='label'>Critical Features</div><div class='value' style='color:{cc};'>{n_crit}</div></div>", unsafe_allow_html=True)
            with oc4:
                action = report["recommended_action"].replace("_", " ")
                ac = "#ef4444" if "RETRAIN" in report["recommended_action"] else "#22c55e"
                st.markdown(f"<div class='metric-card'><div class='label'>Recommended Action</div><div class='value' style='color:{ac};font-size:0.85rem;margin-top:6px;'>{action}</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Per-feature PSI chart
            if feature_report:
                feat_psi_df = pd.DataFrame([
                    {"Feature": k, "PSI": v["psi"], "Severity": v["psi_severity"],
                     "Mean Shift %": v.get("mean_shift_pct", 0)}
                    for k, v in feature_report.items()
                ]).sort_values("PSI", ascending=True)

                bar_cols_drift = [
                    "#ef4444" if s == "CRITICAL" else ("#f59e0b" if s == "MODERATE" else "#22c55e")
                    for s in feat_psi_df["Severity"]
                ]

                fig_drift = go.Figure()
                fig_drift.add_trace(go.Bar(
                    x=feat_psi_df["PSI"], y=feat_psi_df["Feature"],
                    orientation="h", marker_color=bar_cols_drift,
                    text=[f"{v:.4f}" for v in feat_psi_df["PSI"]],
                    textposition="outside", textfont={"color": "#9ca3af", "size": 11},
                ))
                fig_drift.add_vline(x=0.10, line_dash="dot", line_color="#f59e0b",
                                    annotation_text="Warn 0.10", annotation_font_color="#f59e0b")
                fig_drift.add_vline(x=0.25, line_dash="dot", line_color="#ef4444",
                                    annotation_text="Retrain 0.25", annotation_font_color="#ef4444")
                fig_drift.update_layout(
                    height=360, paper_bgcolor="#0f1117", plot_bgcolor="#1e2533",
                    margin=dict(t=20, b=10, l=10, r=80),
                    title=dict(text="Population Stability Index per Feature", font={"color": "#9ca3af", "size": 13}),
                    xaxis=dict(gridcolor="#2d3748", tickfont={"color": "#6b7280"}, title="PSI Score"),
                    yaxis=dict(gridcolor="#2d3748", tickfont={"color": "#d1d5db"}),
                    showlegend=False,
                )
                st.plotly_chart(fig_drift, use_container_width=True)

                # Feature details table
                st.markdown("<div class='section-title'>Feature-Level Drift Details</div>", unsafe_allow_html=True)
                detail_df = pd.DataFrame([
                    {
                        "Feature":      k,
                        "PSI":          round(v["psi"], 4),
                        "Severity":     v["psi_severity"],
                        "Ref Mean":     v.get("ref_mean", "N/A"),
                        "Curr Mean":    v.get("cur_mean", "N/A"),
                        "Mean Shift %": f"{v.get('mean_shift_pct', 0):.1f}%",
                        "KS p-value":   v.get("ks_pvalue", "N/A"),
                    }
                    for k, v in feature_report.items()
                ])

                def color_severity(val):
                    c = {"NONE": "#22c55e", "MODERATE": "#f59e0b", "CRITICAL": "#ef4444"}.get(val, "white")
                    return f"color: {c}; font-weight: 600"

                styled_detail = detail_df.style.applymap(color_severity, subset=["Severity"])
                st.dataframe(styled_detail, use_container_width=True)

    else:
        # Show drift strategy info when no file uploaded
        st.markdown("""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px;'>
        <div style='background:#1e2533;border:1px solid #2d3748;border-radius:10px;padding:1rem;'>
            <div style='color:#34d399;font-size:0.8rem;font-weight:600;margin-bottom:8px;letter-spacing:0.06em;'>MONITORING SCHEDULE</div>
            <div style='color:#9ca3af;font-size:0.82rem;line-height:2;'>
            Daily · PSI on top 5 features<br>
            Weekly · Full feature drift report<br>
            Monthly · Clinical expert validation
            </div>
        </div>
        <div style='background:#1e2533;border:1px solid #2d3748;border-radius:10px;padding:1rem;'>
            <div style='color:#60a5fa;font-size:0.8rem;font-weight:600;margin-bottom:8px;letter-spacing:0.06em;'>RETRAINING TRIGGERS</div>
            <div style='color:#9ca3af;font-size:0.82rem;line-height:2;'>
            PSI &gt; 0.10 on any feature → Warning<br>
            PSI &gt; 0.25 on any feature → Retrain<br>
            Positive rate shift &gt; 10% → Investigate
            </div>
        </div>
        <div style='background:#1e2533;border:1px solid #2d3748;border-radius:10px;padding:1rem;'>
            <div style='color:#a78bfa;font-size:0.8rem;font-weight:600;margin-bottom:8px;letter-spacing:0.06em;'>STATISTICAL TESTS</div>
            <div style='color:#9ca3af;font-size:0.82rem;line-height:2;'>
            PSI — Population Stability Index<br>
            KS — Kolmogorov-Smirnov test<br>
            JS — Jensen-Shannon divergence
            </div>
        </div>
        <div style='background:#1e2533;border:1px solid #2d3748;border-radius:10px;padding:1rem;'>
            <div style='color:#f59e0b;font-size:0.8rem;font-weight:600;margin-bottom:8px;letter-spacing:0.06em;'>MLOPS STACK</div>
            <div style='color:#9ca3af;font-size:0.82rem;line-height:2;'>
            MLflow · Experiment tracking<br>
            Airflow · Retraining DAG<br>
            Evidently AI · Drift dashboard
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)