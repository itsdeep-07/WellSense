import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Base directory (works both locally and on Streamlit Cloud) ────────────────
BASE_DIR = Path(__file__).parent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WellSense — Student Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        text-align: center;
    }
    .metric-card h2 { font-size: 2rem; font-weight: 700; color: #667eea; margin: 0; }
    .metric-card p  { color: #666; margin: 0.2rem 0 0 0; font-size: 0.9rem; }

    .result-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 1.5rem; border-radius: 12px; text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white; padding: 1.5rem; border-radius: 12px; text-align: center;
    }
    .result-high h2, .result-low h2 { font-size: 1.8rem; margin: 0; }
    .result-high p, .result-low p   { opacity: 0.9; margin: 0.4rem 0 0 0; }

    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600; width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }

    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #2d3436;
        border-bottom: 3px solid #667eea; padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load saved model artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load(BASE_DIR / 'models' / 'best_model.pkl')
        scaler   = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
        encoders = joblib.load(BASE_DIR / 'models' / 'label_encoders.pkl')
        results  = joblib.load(BASE_DIR / 'models' / 'results_df.pkl')
        return model, scaler, encoders, results, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, None, False


model, scaler, label_encoders, results_df, model_loaded = load_artifacts()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.markdown("## 🧠 WellSense")
    st.markdown("*Student Mental Health Predictor*")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Home",
        "🔮 Predict Risk",
        "📊 Model Performance",
        "📘 About"
    ])
    st.divider()
    st.caption("University ML Assessment Project")
    st.caption("Dataset: Kaggle — Shariful07")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 WellSense</h1>
        <p>Student Mental Health Risk Prediction using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>7</h2><p>ML Models Trained</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>101</h2><p>Students in Dataset</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>9</h2><p>Input Features</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>5</h2><p>Syllabus Units</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📌 Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        **WellSense** is a university ML assessment project that predicts student mental health risk
        (depression, anxiety) based on demographic and academic features.

        **Models used:**
        - Logistic Regression
        - Decision Tree
        - Random Forest *(best performer)*
        - Support Vector Machine (SVM)
        - k-Nearest Neighbors
        - Naive Bayes
        - MLP Neural Network
        """)

    with col2:
        st.markdown('<div class="section-title">📂 Dataset Features</div>', unsafe_allow_html=True)
        features_info = pd.DataFrame({
            'Feature': ['Gender', 'Age', 'Course', 'Year of Study', 'CGPA',
                        'Marital Status', 'Anxiety', 'Panic Attack', 'Sought Treatment'],
            'Type': ['Categorical', 'Numerical', 'Categorical', 'Categorical', 'Categorical',
                     'Categorical', 'Binary', 'Binary', 'Binary']
        })
        st.dataframe(features_info, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Risk":
    st.markdown("## 🔮 Predict Depression Risk")
    st.markdown("Fill in the student details below to get a mental health risk prediction.")

    if not model_loaded:
        st.warning("⚠️ Model not found. Please run the Jupyter Notebook first to train and save the model.")
        st.info("Run: `notebooks/WellSense_ML_Pipeline.ipynb` → all cells → model saved to `models/`")
        st.stop()

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Info**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        age    = st.slider("Age", 18, 30, 21)
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])

    with col2:
        st.markdown("**🎓 Academic Info**")
        course = st.selectbox("Course / Major", [
            "Engineering", "BIT", "Laws", "Pendidikan Islam",
            "BCS", "Human Sciences", "Economics", "Nursing",
            "KENMS", "Psychology", "Accounting", "Communication",
            "Marine Science", "KIRKHS", "Biomedical Science"
        ])
        year = st.selectbox("Year of Study", ["year 1", "year 2", "year 3", "year 4"])
        cgpa = st.selectbox("CGPA Range", ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99",
                                            "3.00 - 3.49", "3.50 - 4.00"])

    with col3:
        st.markdown("**🧠 Mental Health Indicators**")
        anxiety     = st.radio("Do you have Anxiety?",     ["No", "Yes"], horizontal=True)
        panic_attack= st.radio("Do you have Panic Attacks?",["No", "Yes"], horizontal=True)
        treatment   = st.radio("Sought Specialist Treatment?", ["No", "Yes"], horizontal=True)

    st.divider()
    predict_btn = st.button("🔮 Predict Mental Health Risk", use_container_width=True)

    if predict_btn:
        # Build input dataframe
        input_data = pd.DataFrame({
            'gender'        : [gender],
            'age'           : [age],
            'course'        : [course],
            'year'          : [year],
            'cgpa'          : [cgpa],
            'marital_status': [marital_status],
            'anxiety'       : [anxiety],
            'panic_attack'  : [panic_attack],
            'treatment'     : [treatment]
        })

        # Encode categorical columns
        cat_cols = ['gender', 'course', 'year', 'cgpa', 'marital_status',
                    'anxiety', 'panic_attack', 'treatment']
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                val = input_data[col].values[0]
                if val in le.classes_:
                    input_data[col] = le.transform([val])
                else:
                    input_data[col] = 0

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        risk_pct = probability[1] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h2>⚠️ High Depression Risk</h2>
                    <p>Risk Score: <strong>{risk_pct:.1f}%</strong></p>
                    <p>This student shows indicators of depression risk.<br>
                    Consider connecting with a counselor or mental health support.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h2>✅ Low Depression Risk</h2>
                    <p>Risk Score: <strong>{risk_pct:.1f}%</strong></p>
                    <p>This student shows low indicators of depression risk.<br>
                    Keep maintaining healthy habits and routines!</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Probability bar
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(7, 1.5))
            ax.barh(['Risk Level'], [probability[1]], color='#ee5a24', height=0.5, label='Depression Risk')
            ax.barh(['Risk Level'], [probability[0]], left=probability[1],
                    color='#00b894', height=0.5, label='No Risk')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.legend(loc='lower right', fontsize=9)
            ax.set_title('Risk Probability Breakdown', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        st.info("⚠️ **Disclaimer:** This tool is for educational purposes only and should not replace professional mental health assessment.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Dashboard")

    if not model_loaded or results_df is None:
        st.warning("⚠️ Results not found. Please run the Jupyter Notebook to generate model results.")
        st.stop()

    st.markdown("### 📋 Model Comparison Table")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#d4edda')
                 .highlight_min(axis=0, color='#f8d7da'),
                 use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Accuracy Ranking")
        fig, ax = plt.subplots(figsize=(8, 5))
        acc_data = results_df['Accuracy'].astype(float).sort_values()
        colors = ['#667eea' if x == acc_data.max() else '#b2bec3' for x in acc_data]
        ax.barh(acc_data.index, acc_data.values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Accuracy')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        for i, v in enumerate(acc_data.values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### 🎯 F1 Score Ranking")
        fig, ax = plt.subplots(figsize=(8, 5))
        f1_data = results_df['F1 Score'].astype(float).sort_values()
        colors = ['#00b894' if x == f1_data.max() else '#b2bec3' for x in f1_data]
        ax.barh(f1_data.index, f1_data.values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('F1 Score')
        ax.set_title('Model F1 Score Comparison', fontweight='bold')
        for i, v in enumerate(f1_data.values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    st.markdown("### 📈 Pre-generated Charts")
    st.info("After running the notebook, charts will be saved in the `assets/` folder.")

    asset_files = {
        'Target Distributions'  : BASE_DIR / 'assets' / 'target_distributions.png',
        'Demographics'          : BASE_DIR / 'assets' / 'demographics.png',
        'Depression Analysis'   : BASE_DIR / 'assets' / 'depression_analysis.png',
        'Correlation Heatmap'   : BASE_DIR / 'assets' / 'correlation_heatmap.png',
        'Feature Importance'    : BASE_DIR / 'assets' / 'feature_importance.png',
        'Model Comparison'      : BASE_DIR / 'assets' / 'model_comparison.png',
        'ROC-AUC Curves'        : BASE_DIR / 'assets' / 'roc_auc_curves.png',
        'Confusion Matrices'    : BASE_DIR / 'assets' / 'confusion_matrices.png',
    }

    cols = st.columns(2)
    for i, (title, path) in enumerate(asset_files.items()):
        with cols[i % 2]:
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_column_width=True)
            else:
                st.markdown(f"**{title}** *(run notebook to generate)*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📘 About":
    st.markdown("## 📘 About WellSense")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Project Goal
        WellSense aims to identify students at risk of depression or anxiety using
        machine learning, enabling early intervention and support.

        ### 📦 Tech Stack
        - **Python 3.10+**
        - **Scikit-learn** — ML models
        - **Pandas / NumPy** — Data processing
        - **Matplotlib / Seaborn** — Visualisation
        - **Imbalanced-learn** — SMOTE
        - **Streamlit** — Web deployment
        - **Joblib** — Model persistence

        ### 📂 Dataset
        - **Source:** Kaggle — Shariful07
        - **Rows:** 101 students
        - **Features:** 10 columns
        - **Task:** Binary classification (Depression: Yes/No)
        """)

    with col2:
        st.markdown("""
        ### 🗂️ Project Structure
        ```
        WellSense/
        ├── data/
        │   └── Student Mental health.csv
        ├── models/
        │   ├── best_model.pkl
        │   ├── scaler.pkl
        │   └── label_encoders.pkl
        ├── notebooks/
        │   └── WellSense_ML_Pipeline.ipynb
        ├── assets/
        │   └── *.png (generated charts)
        ├── app.py
        ├── requirements.txt
        └── README.md
        ```

        ### 📋 Syllabus Coverage
        | Unit | Topic |
        |------|-------|
        | II  | Data preprocessing, EDA |
        | III | Classification models |
        | IV  | Neural Networks (MLP) |
        | V   | Model evaluation, tuning |
        """)

    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.error("""
    This tool is built **for educational and academic purposes only**.
    It is not a clinical diagnostic tool and should not be used as a substitute
    for professional mental health assessment or treatment.
    If you or someone you know is struggling, please reach out to a qualified mental health professional.
    """)
