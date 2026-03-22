# 🧠 WellSense — Student Mental Health Risk Prediction

> A machine learning system for predicting student depression risk, built with an India-relevant dataset and deployed as a real-time web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wellsense.streamlit.app)

---

## 📌 Overview

**WellSense** is a classification system that predicts student depression risk using seven machine learning algorithms. Trained on 27,450 India-relevant student records with features like academic pressure, financial stress, and sleep patterns.

---

## 🗂️ Project Structure

```
WellSense/
├── data/
│   └── student_depression.csv       ← Download from Kaggle
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── results_df.pkl
│   └── features.pkl
├── assets/                          ← Generated charts
├── WellSense_ML_Pipeline.ipynb      ← Full ML pipeline
├── app.py                           ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

| Detail | Info |
|--------|------|
| Source | [Kaggle — adilshamim8](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset) |
| Records | 27,450 students |
| Context | India — 0–10 academic scale, Indian cities, Indian degrees |
| Target | Depression (0 = No, 1 = Yes) |

---

## 🤖 Models Trained

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest (Tuned) | 0.8626 | 0.8626 | 0.9377 |
| Random Forest | 0.8627 | 0.8627 | 0.9370 |
| Logistic Regression | 0.8566 | 0.8566 | 0.9311 |
| SVM | 0.8549 | 0.8549 | 0.9240 |
| Naive Bayes | 0.8500 | 0.8499 | 0.9282 |
| MLP Neural Network | 0.8454 | 0.8454 | 0.9169 |
| k-NN | 0.8364 | 0.8364 | 0.9007 |
| Decision Tree | 0.7965 | 0.7965 | 0.7965 |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/itsdeep-07/WellSense.git
cd WellSense
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
- Go to: https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset
- Save as `data/student_depression.csv`

### 4. Run the notebook
```bash
jupyter notebook WellSense_ML_Pipeline.ipynb
```
Run all cells — models save to `models/` automatically.

### 5. Launch the app
```bash
streamlit run app.py
```

---

## 📊 Pipeline

| Phase | Description |
|-------|-------------|
| EDA | Distributions, correlations, feature analysis |
| Preprocessing | Label encoding, StandardScaler, SMOTE, PCA |
| Training | 7 classifiers trained and compared |
| Evaluation | Accuracy, F1, ROC-AUC, Confusion Matrix |
| Optimisation | GridSearchCV, 5-Fold Cross Validation |

---

## 🌐 App Pages

- **Dashboard** — Overview, leaderboard, quick prediction
- **Model Metrics** — Full performance table, confusion matrix, feature importance
- **Predict** — 14-feature form with real-time risk assessment
- **About** — Project details and tech stack

---

## 📋 Tech Stack

`Python` `scikit-learn` `Pandas` `NumPy` `SMOTE` `Streamlit` `Joblib` `Matplotlib` `Seaborn`

---

## 👥 Authors

| Name | Student ID |
|------|-----------|
| Deepak Kumar | 2024UCS1677 |
| Piyush Kumar | 2024UCS1697 |
| Varun Yadav | 2024UCS1669 |

**Netaji Subhas University of Technology, New Delhi**
Department of Computer Science Engineering

---

## ⚠️ Disclaimer

For educational purposes only. Not a clinical diagnostic tool or substitute for professional mental health assessment.
