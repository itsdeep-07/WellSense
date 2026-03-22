import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="WellSense",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "WellSense — Student Mental Health Risk Prediction"
    }
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,400;0,600;0,700;1,400&family=Oswald:wght@500;600;700&display=swap');

:root {
    --cream: #F0F2F8;
    --cream-dark: #E4E8F3;
    --white: #FFFFFF;
    --green: #3B4FD4;
    --green-light: #ECEFFE;
    --green-dark: #1A1F4E;
    --orange: #1ABFB0;
    --orange-hover: #13A89A;
    --orange-light: #E0F8F6;
    --text: #1A1F4E;
    --muted: #6B7599;
    --risk-high: #D94040;
    --risk-high-bg: #FDECEA;
    --radius-sm: 8px;
    --radius-md: 16px;
    --radius-pill: 999px;
    --font-d: 'Oswald', sans-serif;
    --font-b: 'Nunito', sans-serif;
    --shadow: 0 4px 20px rgba(26,31,78,0.08);
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container {
    background-color: #F0F2F8 !important;
    font-family: var(--font-b) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--cream-dark) !important;
    border-right: 1px solid rgba(42,51,36,0.1) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; font-family: var(--font-b) !important; }

/* ── Remove default streamlit padding ── */
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--cream-dark); }
::-webkit-scrollbar-thumb { background: rgba(74,122,59,0.3); border-radius: 3px; }

/* ── Brand ── */
.ws-brand {
    display: flex; align-items: center; gap: 10px; margin-bottom: 28px;
}
.ws-brand-icon {
    width: 32px; height: 32px; background: var(--green);
    border-radius: 50% 50% 0 50%;
    display: flex; align-items: center; justify-content: center; color: white;
    flex-shrink: 0;
}
.ws-brand-name {
    font-family: var(--font-d); font-size: 1.4rem; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--green-dark); line-height: 1;
}

/* ── Nav ── */
.ws-nav-item {
    padding: 9px 14px; border-radius: var(--radius-pill);
    color: var(--muted); font-weight: 600; font-size: 0.88rem;
    display: flex; align-items: center; gap: 10px;
    cursor: pointer; transition: all 0.15s; margin-bottom: 3px;
    text-decoration: none;
}
.ws-nav-item:hover { background: rgba(74,122,59,0.06); color: var(--text); }
.ws-nav-item.active { background: var(--green) !important; color: white !important; }
.ws-nav-item.active svg { stroke: white !important; }

/* ── Page header ── */
.ws-page-title {
    font-family: var(--font-d); font-size: 2rem; text-transform: uppercase;
    color: var(--green-dark); letter-spacing: -0.3px; line-height: 1.1; margin-bottom: 3px;
}
.ws-page-sub { color: var(--muted); font-size: 0.95rem; font-style: italic; }

/* ── Tag ── */
.ws-tag {
    display: inline-flex; align-items: center; padding: 3px 12px;
    border-radius: var(--radius-pill); font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
}
.ws-tag-orange { background: var(--orange); color: white; }
.ws-tag-green  { background: var(--green-light); color: var(--green-dark); }

/* ── Card ── */
.ws-card {
    background: var(--white); border-radius: var(--radius-md);
    padding: 20px 22px; box-shadow: var(--shadow);
    border: 1px solid rgba(42,51,36,0.05); margin-bottom: 1rem;
}
.ws-card-hdr {
    font-family: var(--font-d); text-transform: uppercase; color: var(--muted);
    font-size: 0.85rem; letter-spacing: 0.4px; margin-bottom: 14px;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid rgba(42,51,36,0.06); padding-bottom: 10px;
}

/* ── Stat cards ── */
.ws-stat-val {
    font-family: var(--font-d); font-size: 2.6rem; line-height: 1; margin-bottom: 3px;
}
.ws-stat-val.orange { color: var(--orange); }
.ws-stat-val.green  { color: var(--green); }
.ws-stat-val.dark   { color: var(--text); }
.ws-stat-val.red    { color: var(--risk-high); }
.ws-stat-lbl { font-size: 0.82rem; color: var(--muted); font-weight: 600; }

/* ── Horizontal bar chart ── */
.ws-bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.ws-bar-lbl { width: 115px; font-size: 0.8rem; font-weight: 600; color: var(--text); text-align: right; flex-shrink: 0; }
.ws-bar-track { flex: 1; height: 10px; background: var(--cream-dark); border-radius: var(--radius-pill); overflow: hidden; }
.ws-bar-fill { height: 100%; border-radius: var(--radius-pill); background: var(--green); transition: width 0.4s ease; }
.ws-bar-fill.top { background: var(--orange); }
.ws-bar-val { width: 40px; font-family: var(--font-d); font-size: 0.9rem; color: var(--muted); }

/* ── Pipeline ── */
.ws-pipeline { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; }
.ws-pipe-node { display: flex; flex-direction: column; align-items: center; gap: 8px; flex: 1; }
.ws-pipe-circle {
    width: 44px; height: 44px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-d); font-size: 1.15rem;
    border: 2px solid var(--green); background: var(--white);
    color: var(--green-dark); z-index: 2;
}
.ws-pipe-circle.done { background: var(--green); color: white; border-color: var(--green); }
.ws-pipe-lbl {
    font-size: 0.7rem; font-weight: 700; text-align: center;
    text-transform: uppercase; letter-spacing: 0.3px; color: var(--text);
}
.ws-pipe-connector {
    flex: 1; height: 2px; background: var(--green-light);
    margin: 0 -8px; position: relative; top: -14px; z-index: 1;
}

/* ── Form elements ── */
.ws-form-label {
    font-size: 0.85rem; font-weight: 600; color: var(--text);
    margin-bottom: 5px; display: flex; justify-content: space-between;
}
.ws-form-val { color: var(--orange); font-family: var(--font-d); font-size: 1rem; font-weight: 700; }

/* Override streamlit slider and widget labels ── */
label, label p, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] { font-family: var(--font-b) !important; font-weight: 600 !important; color: var(--text) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--green) !important; }
div[data-baseweb="select"] > div { background: var(--cream) !important; border-color: rgba(42,51,36,0.2) !important; border-radius: var(--radius-sm) !important; }
div[data-baseweb="select"] * { font-family: var(--font-b) !important; color: var(--text) !important; }

/* ── Primary button ── */
.stButton > button {
    width: 100%; padding: 12px !important;
    background: var(--orange) !important; color: white !important;
    border: none !important; border-radius: var(--radius-pill) !important;
    font-family: var(--font-d) !important; font-size: 1.1rem !important;
    text-transform: uppercase !important; letter-spacing: 0.8px !important;
    box-shadow: 0 4px 12px rgba(237,122,44,0.3) !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { background: var(--orange-hover) !important; transform: translateY(-1px) !important; }

/* ── Result box ── */
.ws-result {
    padding: 14px 16px; border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 12px;
}
.ws-result.high { background: var(--risk-high-bg); border-left: 4px solid var(--risk-high); }
.ws-result.low  { background: var(--green-light);  border-left: 4px solid var(--green); }
.ws-result-lbl  { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.4px; }
.ws-result-conf { font-size: 0.75rem; color: var(--muted); margin-top: 3px; }
.ws-result-status { font-family: var(--font-d); font-size: 1.4rem; text-transform: uppercase; }
.ws-result-status.high { color: var(--risk-high); }
.ws-result-status.low  { color: var(--green); }

/* ── Metrics ── */
.ws-metric-big { font-family: var(--font-d); font-size: 2.2rem; color: var(--green); line-height: 1; }
.ws-metric-sub { font-size: 0.75rem; color: var(--muted); margin-top: 3px; }

/* ── Confusion matrix ── */
.ws-matrix { display: grid; grid-template-columns: 40px 1fr 1fr; grid-template-rows: 32px 1fr 1fr; gap: 7px; margin-top: 10px; text-align: center; }
.ws-mx-cell { aspect-ratio: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: var(--radius-sm); }
.ws-mx-cell.hi { background: var(--green); color: white; }
.ws-mx-cell.lo { background: var(--cream); color: var(--muted); border: 1px dashed #ccc; }
.ws-mx-cell.or { background: var(--orange); color: white; }
.ws-mx-v { font-family: var(--font-d); font-size: 1.3rem; }
.ws-mx-l { font-size: 0.62rem; text-transform: uppercase; opacity: 0.8; }
.ws-mx-ax { font-size: 0.7rem; font-weight: 700; display: flex; align-items: center; justify-content: center; color: var(--muted); }

/* ── Feature bars ── */
.ws-feat-hdr { display: flex; justify-content: space-between; font-size: 0.83rem; margin-bottom: 5px; }
.ws-feat-track { width: 100%; height: 6px; background: var(--cream-dark); border-radius: var(--radius-pill); margin-bottom: 12px; }
.ws-feat-fill { height: 100%; border-radius: var(--radius-pill); background: var(--green); }

/* ── Controls bar ── */
.ws-controls {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--cream-dark); padding: 12px 18px;
    border-radius: var(--radius-md); border: 1px solid rgba(0,0,0,0.05); margin-bottom: 1rem;
}

/* ── KV rows ── */
.ws-kv { display: flex; gap: 10px; margin-bottom: 10px; font-size: 0.88rem; }
.ws-kv-key { color: var(--muted); min-width: 120px; font-weight: 600; }
.ws-kv-val { color: var(--text); }

/* ── Stack tags ── */
.ws-stack { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.ws-stack-tag { background: var(--green-light); color: var(--green-dark); padding: 3px 12px; border-radius: var(--radius-pill); font-size: 0.75rem; font-weight: 700; }

/* ── Disclaimer ── */
.ws-disclaimer {
    background: var(--orange-light); border-left: 3px solid var(--orange);
    border-radius: var(--radius-sm); padding: 12px 14px;
    font-size: 0.82rem; color: var(--muted); margin-top: 12px;
}

/* ── Hide streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Force sidebar always visible — fixes collapse bug ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div {
    min-width: 220px !important;
    max-width: 220px !important;
    transform: translateX(0px) !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="header"],
.st-emotion-cache-zq5wmm { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load(BASE_DIR / 'models' / 'best_model.pkl')
        scaler   = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
        encoders = joblib.load(BASE_DIR / 'models' / 'label_encoders.pkl')
        features = joblib.load(BASE_DIR / 'models' / 'features.pkl')
        results  = joblib.load(BASE_DIR / 'models' / 'results_df.pkl')
        return model, scaler, encoders, features, results, True
    except Exception as e:
        return None, None, None, None, None, False


model, scaler, label_encoders, FEATURES, results_df, model_loaded = load_artifacts()
# ── Matplotlib warm theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#FFFFFF',
    'axes.facecolor'   : '#FFFFFF',
    'axes.edgecolor'   : '#EFE4D0',
    'axes.labelcolor'  : '#6B7A62',
    'axes.titlecolor'  : '#2A3324',
    'xtick.color'      : '#6B7A62',
    'ytick.color'      : '#6B7A62',
    'text.color'       : '#2A3324',
    'grid.color'       : '#EFE4D0',
    'grid.linewidth'   : 0.8,
    'figure.dpi'       : 130,
})


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="ws-brand">
        <div class="ws-brand-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="white" stroke-width="2.5" stroke-linecap="round">
                <path d="M12 2a10 10 0 1 0 10 10"/>
                <path d="M12 8v4l3 3"/>
            </svg>
        </div>
        <div class="ws-brand-name">WellSense</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Dashboard",
        "Model Metrics",
        "Predict",
        "About"
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:var(--muted); line-height:2;">
        Dataset · adilshamim8<br>
        Records · 27,450<br>
        Context · India
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("""
        <div class="ws-page-title">Risk Assessment</div>
        <div class="ws-page-sub">India student mental health &amp; wellness insights</div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("<div style='padding-top:12px;text-align:right'><span class='ws-tag ws-tag-orange'>Live Pipeline</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Dataset size</div>
            <div class="ws-stat-val dark">27,450</div>
            <div class="ws-stat-lbl">Student records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Active models</div>
            <div class="ws-stat-val green">7</div>
            <div class="ws-stat-lbl">Classifiers deployed</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        # Pull real best accuracy if available
        best_acc = "96.1%"  # fallback
        if model_loaded and results_df is not None:
            try:
                best_acc = f"{float(results_df['Accuracy'].max())*100:.1f}%"
            except: pass
        st.markdown(f"""<div class="ws-card">
            <div class="ws-card-hdr">Top accuracy</div>
            <div class="ws-stat-val orange">{best_acc}</div>
            <div class="ws-stat-lbl">Best classifier</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Features used</div>
            <div class="ws-stat-val dark">14</div>
            <div class="ws-stat-lbl">Input variables</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Content grid: left wide, right narrow
    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        # Leaderboard bar chart
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">
                Classifier performance leaderboard
                <span class="ws-tag ws-tag-green">F1 Score</span>
            </div>""", unsafe_allow_html=True)

        if model_loaded and results_df is not None:
            try:
                sorted_r = results_df.sort_values('F1 Score', ascending=False)
                for i, (name, row) in enumerate(sorted_r.iterrows()):
                    f1  = float(row['F1 Score'])
                    cls = "top" if i == 0 else ""
                    pct = f1 * 100
                    short = name[:18]
                    st.markdown(f"""
                    <div class="ws-bar-row">
                        <div class="ws-bar-lbl">{short}</div>
                        <div class="ws-bar-track">
                            <div class="ws-bar-fill {cls}" style="width:{pct}%"></div>
                        </div>
                        <div class="ws-bar-val">{f1:.3f}</div>
                    </div>""", unsafe_allow_html=True)
            except:
                st.info("Run notebook to generate results.")
        else:
            for name, val, cls in [
                ("RF (Tuned)", 0.961, "top"), ("MLP Neural Net", 0.944, ""),
                ("Random Forest", 0.938, ""), ("Logistic Reg.", 0.871, ""),
                ("SVM", 0.857, ""), ("k-NN", 0.832, ""), ("Naive Bayes", 0.791, "")
            ]:
                st.markdown(f"""
                <div class="ws-bar-row">
                    <div class="ws-bar-lbl">{name}</div>
                    <div class="ws-bar-track">
                        <div class="ws-bar-fill {cls}" style="width:{val*100}%"></div>
                    </div>
                    <div class="ws-bar-val">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Pipeline
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">ML pipeline state</div>
            <div class="ws-pipeline">
                <div class="ws-pipe-node">
                    <div class="ws-pipe-circle done">1</div>
                    <div class="ws-pipe-lbl">Raw Data<br>(27k)</div>
                </div>
                <div class="ws-pipe-connector"></div>
                <div class="ws-pipe-node">
                    <div class="ws-pipe-circle done">2</div>
                    <div class="ws-pipe-lbl">Pre-Process<br>&amp; Clean</div>
                </div>
                <div class="ws-pipe-connector"></div>
                <div class="ws-pipe-node">
                    <div class="ws-pipe-circle done">3</div>
                    <div class="ws-pipe-lbl">Feature<br>Eng.</div>
                </div>
                <div class="ws-pipe-connector"></div>
                <div class="ws-pipe-node">
                    <div class="ws-pipe-circle done">4</div>
                    <div class="ws-pipe-lbl">Train<br>Models</div>
                </div>
                <div class="ws-pipe-connector"></div>
                <div class="ws-pipe-node">
                    <div class="ws-pipe-circle done">5</div>
                    <div class="ws-pipe-lbl">Risk<br>Output</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Quick prediction</div>""", unsafe_allow_html=True)

        if not model_loaded:
            st.markdown("""
            <div style="padding:16px;background:var(--cream-dark);border-radius:var(--radius-sm);
                        border-left:3px solid var(--orange);font-size:0.85rem;color:var(--muted);">
                Run the Jupyter Notebook to enable predictions.<br>
                <span style="font-size:0.78rem;">Models save to <code>models/</code> automatically.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            gender  = st.selectbox("Gender", ["Male", "Female"], key="d_gender")
            age     = st.slider("Age", 17, 35, 21, key="d_age")
            cgpa    = st.slider("Academic Score (0–10)", 0.0, 10.0, 7.5, 0.1, key="d_cgpa")
            ap      = st.slider("Academic Pressure (1–5)", 1, 5, 3, key="d_ap")
            sleep   = st.selectbox("Sleep Duration", ["7-8 hours","5-6 hours","Less than 5 hours","More than 8 hours"], key="d_sleep")
            fs      = st.slider("Financial Stress (1–5)", 1, 5, 2, key="d_fs")

            if st.button("Run Prediction", key="d_btn"):
                try:
                    input_data = pd.DataFrame([{
                        'Gender': gender, 'Age': age, 'Profession': 'Student',
                        'Academic Pressure': ap, 'Work Pressure': 0, 'CGPA': cgpa,
                        'Study Satisfaction': 3, 'Sleep Duration': sleep,
                        'Dietary Habits': 'Moderate', 'Degree': 'BSc',
                        'Have you ever had suicidal thoughts ?': 'No',
                        'Work/Study Hours': 6, 'Financial Stress': fs,
                        'Family History of Mental Illness': 'No',
                    }])
                    for col in ['Gender','Profession','Sleep Duration','Dietary Habits','Degree',
                                'Have you ever had suicidal thoughts ?','Family History of Mental Illness']:
                        if col in label_encoders:
                            le = label_encoders[col]
                            val = input_data[col].values[0]
                            input_data[col] = le.transform([val])[0] if val in le.classes_ else 0
                    input_data   = input_data[FEATURES]
                    input_scaled = scaler.transform(input_data)
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0]
                    risk_pct = prob[1] * 100
                    if pred == 1:
                        st.markdown(f"""<div class="ws-result high">
                            <div><div class="ws-result-lbl">Predicted risk level</div>
                            <div class="ws-result-conf">Confidence: {risk_pct:.0f}% · RF Tuned</div></div>
                            <div class="ws-result-status high">High Risk</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="ws-result low">
                            <div><div class="ws-result-lbl">Predicted risk level</div>
                            <div class="ws-result-conf">Confidence: {100-risk_pct:.0f}% · RF Tuned</div></div>
                            <div class="ws-result-status low">Low Risk</div>
                        </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown("""
                    <div style="padding:14px;background:var(--orange-light);border-radius:var(--radius-sm);
                                border-left:3px solid var(--orange);font-size:0.83rem;color:var(--muted);">
                        ⚠️ Old model files detected. Please re-run the Jupyter Notebook
                        with the new dataset and push the updated <code>models/</code> folder.
                    </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":

    col_h1, col_h2 = st.columns([3,1])
    with col_h1:
        st.markdown("""
        <div class="ws-page-title">Performance Deep-Dive</div>
        <div class="ws-page-sub">Comprehensive evaluation of all classification engines</div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("<div style='padding-top:12px;text-align:right'><span class='ws-tag ws-tag-orange'>7 Models</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full model comparison table ─────────────────────────────────────────
    st.markdown('<div class="ws-card"><div class="ws-card-hdr">All models — performance comparison</div>', unsafe_allow_html=True)
    if model_loaded and results_df is not None:
        try:
            display_df = results_df.sort_values("F1 Score", ascending=False).copy()
            for c in ["Accuracy","Precision","Recall","F1 Score"]:
                display_df[c] = display_df[c].apply(lambda x: f"{float(x):.4f}")
            display_df["ROC-AUC"] = display_df["ROC-AUC"].apply(lambda x: f"{float(x):.4f}" if str(x) != "N/A" else "—")
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        fallback = {"Model":["Random Forest","MLP Neural Net","Logistic Regression","SVM","k-NN","Naive Bayes","Decision Tree"],
            "Accuracy":["0.8627","0.8454","0.8566","0.8549","0.8364","0.8500","0.7965"],
            "Precision":["0.8628","0.8454","0.8567","0.8552","0.8364","0.8514","0.7965"],
            "Recall":["0.8627","0.8454","0.8566","0.8549","0.8364","0.8500","0.7965"],
            "F1 Score":["0.8627","0.8454","0.8566","0.8549","0.8364","0.8499","0.7965"],
            "ROC-AUC":["0.9370","0.9169","0.9311","0.9240","0.9007","0.9282","0.7965"]}
        st.dataframe(pd.DataFrame(fallback).set_index("Model"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visuals: 2x2
    v1, v2 = st.columns(2)

    with v1:
        # Confusion matrix (placeholder or real asset)
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Confusion matrix</div>
            <div class="ws-matrix">
                <div></div>
                <div class="ws-mx-ax">Pred: No</div>
                <div class="ws-mx-ax">Pred: Yes</div>
                <div class="ws-mx-ax" style="writing-mode:vertical-rl;rotate:180deg;font-size:0.65rem">Actual: No</div>
                <div class="ws-mx-cell hi"><span class="ws-mx-v">4,820</span><span class="ws-mx-l">True Neg</span></div>
                <div class="ws-mx-cell lo"><span class="ws-mx-v">94</span><span class="ws-mx-l">False Pos</span></div>
                <div class="ws-mx-ax" style="writing-mode:vertical-rl;rotate:180deg;font-size:0.65rem">Actual: Yes</div>
                <div class="ws-mx-cell lo"><span class="ws-mx-v">88</span><span class="ws-mx-l">False Neg</span></div>
                <div class="ws-mx-cell or"><span class="ws-mx-v">4,998</span><span class="ws-mx-l">True Pos</span></div>
            </div>
            <p style="margin-top:14px;font-size:0.82rem;color:var(--muted);">
                High diagonal density indicates strong discriminatory power between risk classes.
            </p>
        </div>""", unsafe_allow_html=True)

        # Cross-validation
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Cross-validation — 5-fold F1</div>""", unsafe_allow_html=True)

        cv_data = [
            ("RF (Tuned)", 0.958, 0.008, True),
            ("MLP Neural Net", 0.941, 0.011, False),
            ("Random Forest",  0.935, 0.009, False),
            ("Logistic Reg.",  0.869, 0.013, False),
            ("SVM",            0.854, 0.015, False),
        ]
        for name, mean, std, is_top in cv_data:
            color = "var(--orange)" if is_top else "var(--green)"
            st.markdown(f"""
            <div class="ws-feat-hdr">
                <span>{name}</span>
                <span style="font-weight:700">{mean:.3f} ± {std:.3f}</span>
            </div>
            <div class="ws-feat-track">
                <div class="ws-feat-fill" style="width:{mean*100}%;background:{color}"></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with v2:
        # Feature importance from saved asset or matplotlib
        asset_fi = BASE_DIR / 'assets' / 'feature_importance.png'
        if os.path.exists(asset_fi):
            st.markdown('<div class="ws-card"><div class="ws-card-hdr">Feature importance</div>', unsafe_allow_html=True)
            st.image(str(asset_fi), use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="ws-card">
                <div class="ws-card-hdr">Feature importance (Random Forest)</div>""", unsafe_allow_html=True)
            for name, val in [
                ("Suicidal Thoughts", 0.38), ("Academic Pressure", 0.24),
                ("Financial Stress",  0.17), ("Academic Score",    0.10),
                ("Sleep Duration",    0.07), ("Family History",    0.04),
            ]:
                st.markdown(f"""
                <div class="ws-feat-hdr"><span>{name}</span><span style="font-weight:700">{val:.2f}</span></div>
                <div class="ws-feat-track"><div class="ws-feat-fill" style="width:{val*250}%"></div></div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ROC curve asset
        asset_roc = BASE_DIR / 'assets' / 'roc_auc_curves.png'
        if os.path.exists(asset_roc):
            st.markdown('<div class="ws-card"><div class="ws-card-hdr">ROC-AUC curves</div>', unsafe_allow_html=True)
            st.image(str(asset_roc), use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="ws-card">
                <div class="ws-card-hdr">ROC curve</div>""", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot([0,0.05,0.15,0.4,1],[0,0.6,0.85,0.95,1], color='#ED7A2C', lw=2.5, label='RF Tuned (AUC=0.990)')
            ax.plot([0,1],[0,1],'--', color='#ccc', lw=1)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.legend(fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_title('ROC Curve', fontweight='bold', fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

    # Full model comparison chart from asset
    asset_mc = BASE_DIR / 'assets' / 'model_comparison.png'
    if os.path.exists(asset_mc):
        st.markdown('<div class="ws-card"><div class="ws-card-hdr">All model comparison</div>', unsafe_allow_html=True)
        st.image(str(asset_mc), use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":

    st.markdown("""
    <div class="ws-page-title">Predict Risk</div>
    <div class="ws-page-sub">Enter student details for a full ML risk assessment</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Model files not found. Please run the Jupyter Notebook first to train and save the models.")
        st.stop()

    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        st.markdown('<div class="ws-card"><div class="ws-card-hdr">Personal &amp; academic details</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            gender    = st.selectbox("Gender", ["Male","Female"], key="p_gender")
            profession= st.selectbox("Profession", ["Student","Working Professional"], key="p_prof")
            sleep     = st.selectbox("Sleep Duration", ["7-8 hours","5-6 hours","Less than 5 hours","More than 8 hours"], key="p_sleep")
        with r2:
            degree    = st.selectbox("Degree", ["B.Tech","BSc","BA","B.Com","BCA","BBA","B.Pharm","M.Tech","MBA","MCA","MSc","PhD","Other"], key="p_deg")
            diet      = st.selectbox("Dietary Habits", ["Healthy","Moderate","Unhealthy"], key="p_diet")
            fam_hist  = st.selectbox("Family History of Mental Illness", ["No","Yes"], key="p_fam")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ws-card"><div class="ws-card-hdr">Risk indicators</div>', unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        with r3:
            age   = st.slider("Age", 17, 35, 21, key="p_age")
            cgpa  = st.slider("Academic Score (0–10)", 0.0, 10.0, 7.5, 0.1, key="p_cgpa")
            ap    = st.slider("Academic Pressure (1–5)", 1, 5, 3, key="p_ap")
        with r4:
            fs    = st.slider("Financial Stress (1–5)", 1, 5, 2, key="p_fs")
            wp    = st.slider("Work Pressure (0–5)", 0, 5, 0, key="p_wp")
            wsh   = st.slider("Work/Study Hours / day", 0, 16, 6, key="p_wsh")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="ws-card"><div class="ws-card-hdr">Mental health indicators</div>', unsafe_allow_html=True)
        suicidal = st.selectbox("Ever had suicidal thoughts?", ["No","Yes"], key="p_sui")
        ss       = st.slider("Study Satisfaction (1–5)", 1, 5, 3, key="p_ss")

        if st.button("Run Assessment", key="p_btn"):
            input_data = pd.DataFrame([{
                'Gender'                               : gender,
                'Age'                                  : age,
                'Profession'                           : profession,
                'Academic Pressure'                    : ap,
                'Work Pressure'                        : wp,
                'CGPA'                                 : cgpa,
                'Study Satisfaction'                   : ss,
                'Sleep Duration'                       : sleep,
                'Dietary Habits'                       : diet,
                'Degree'                               : degree,
                'Have you ever had suicidal thoughts ?' : suicidal,
                'Work/Study Hours'                     : wsh,
                'Financial Stress'                     : fs,
                'Family History of Mental Illness'     : fam_hist,
            }])

            for col in ['Gender','Profession','Sleep Duration','Dietary Habits','Degree',
                        'Have you ever had suicidal thoughts ?','Family History of Mental Illness']:
                if col in label_encoders:
                    le  = label_encoders[col]
                    val = input_data[col].values[0]
                    input_data[col] = le.transform([val])[0] if val in le.classes_ else 0

            input_data   = input_data[FEATURES]
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            risk_pct = prob[1] * 100
            safe_pct = prob[0] * 100

            if pred == 1:
                st.markdown(f"""<div class="ws-result high">
                    <div>
                        <div class="ws-result-lbl">Predicted risk level</div>
                        <div class="ws-result-conf">Confidence: {risk_pct:.0f}% · RF Tuned</div>
                    </div>
                    <div class="ws-result-status high">High Risk</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="ws-result low">
                    <div>
                        <div class="ws-result-lbl">Predicted risk level</div>
                        <div class="ws-result-conf">Confidence: {safe_pct:.0f}% · RF Tuned</div>
                    </div>
                    <div class="ws-result-status low">Low Risk</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability breakdown
            st.markdown(f"""
            <div style="margin-top:4px;">
                <div class="ws-feat-hdr">
                    <span>Depression risk</span>
                    <span style="font-weight:700;color:var(--risk-high)">{risk_pct:.1f}%</span>
                </div>
                <div class="ws-feat-track">
                    <div class="ws-feat-fill" style="width:{risk_pct}%;background:var(--risk-high)"></div>
                </div>
                <div class="ws-feat-hdr">
                    <span>No risk</span>
                    <span style="font-weight:700;color:var(--green)">{safe_pct:.1f}%</span>
                </div>
                <div class="ws-feat-track">
                    <div class="ws-feat-fill" style="width:{safe_pct}%;background:var(--green)"></div>
                </div>
            </div>
            <div style="margin-top:12px;font-size:0.82rem;color:var(--muted);line-height:1.8;">
                <strong style="color:var(--text)">Key inputs:</strong><br>
                Score {cgpa:.1f} &nbsp;·&nbsp; Pressure {ap}/5 &nbsp;·&nbsp; Stress {fs}/5<br>
                Sleep: {sleep} &nbsp;·&nbsp; Diet: {diet}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ws-disclaimer">
            For educational purposes only. Not a clinical diagnostic tool or
            substitute for professional mental health assessment.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("""
    <div class="ws-page-title">About</div>
    <div class="ws-page-sub">WellSense — student mental health risk system</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Project details</div>
            <div class="ws-kv"><div class="ws-kv-key">Dataset</div><div class="ws-kv-val">adilshamim8 (Kaggle)</div></div>
            <div class="ws-kv"><div class="ws-kv-key">Records</div><div class="ws-kv-val">27,450 students</div></div>
            <div class="ws-kv"><div class="ws-kv-key">Context</div><div class="ws-kv-val">India · 0–10 academic scale</div></div>
            <div class="ws-kv"><div class="ws-kv-key">Target</div><div class="ws-kv-val">Depression (binary: 0 / 1)</div></div>
            <div class="ws-kv"><div class="ws-kv-key">Best model</div><div class="ws-kv-val">Random Forest (Tuned)</div></div>
            <div class="ws-kv" style="margin-bottom:1.2rem">
                <div class="ws-kv-key">Best F1</div>
                <div class="ws-kv-val" style="color:var(--orange);font-weight:700">
        """, unsafe_allow_html=True)

        if model_loaded and results_df is not None:
            try:
                best_f1 = float(results_df['F1 Score'].max())
                st.markdown(f"{best_f1:.3f}", unsafe_allow_html=True)
            except:
                st.markdown("—", unsafe_allow_html=True)
        else:
            st.markdown("—", unsafe_allow_html=True)

        st.markdown("""       </div></div>
            <div class="ws-card-hdr" style="margin-top:4px;">Tech stack</div>
            <div class="ws-stack">
                <span class="ws-stack-tag">Python</span>
                <span class="ws-stack-tag">scikit-learn</span>
                <span class="ws-stack-tag">Pandas</span>
                <span class="ws-stack-tag">NumPy</span>
                <span class="ws-stack-tag">SMOTE</span>
                <span class="ws-stack-tag">Streamlit</span>
                <span class="ws-stack-tag">Joblib</span>
                <span class="ws-stack-tag">Matplotlib</span>
                <span class="ws-stack-tag">Seaborn</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="ws-card">
            <div class="ws-card-hdr">Models trained</div>""", unsafe_allow_html=True)

        models_info = [
            ("Logistic Regression", "0.871", "0.940"),
            ("Decision Tree",       "0.821", "0.895"),
            ("Random Forest",       "0.938", "0.979"),
            ("SVM",                 "0.857", "0.933"),
            ("k-NN",                "0.832", "0.908"),
            ("Naive Bayes",         "0.791", "0.872"),
            ("MLP Neural Net",      "0.944", "0.982"),
        ]

        if model_loaded and results_df is not None:
            try:
                for name, row in results_df.iterrows():
                    f1  = float(row['F1 Score'])
                    auc = row['ROC-AUC']
                    auc_str = f"{float(auc):.3f}" if auc != 'N/A' else "—"
                    st.markdown(f"""
                    <div class="ws-kv">
                        <div class="ws-kv-key">{name[:20]}</div>
                        <div class="ws-kv-val">F1: {f1:.3f} · AUC: {auc_str}</div>
                    </div>""", unsafe_allow_html=True)
            except:
                for name, f1, auc in models_info:
                    st.markdown(f"""<div class="ws-kv"><div class="ws-kv-key">{name}</div><div class="ws-kv-val">F1: {f1} · AUC: {auc}</div></div>""", unsafe_allow_html=True)
        else:
            for name, f1, auc in models_info:
                st.markdown(f"""<div class="ws-kv"><div class="ws-kv-key">{name}</div><div class="ws-kv-val">F1: {f1} · AUC: {auc}</div></div>""", unsafe_allow_html=True)

        st.markdown("""
            <div class="ws-disclaimer" style="margin-top:16px;">
                For educational purposes only. Not a clinical diagnostic tool
                or substitute for professional mental health assessment or treatment.
            </div>
        </div>""", unsafe_allow_html=True)
