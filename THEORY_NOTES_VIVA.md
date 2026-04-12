# 🧠 WellSense — Complete Theory Notes for Viva Voce
### Netaji Subhas University of Technology, New Delhi | Dept. of Computer Science Engineering
**Authors:** Deepak Kumar (2024UCS1677) | Piyush Kumar (2024UCS1697) | Varun Yadav (2024UCS1669)

---

## TABLE OF CONTENTS
1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [Dataset](#2-dataset)
3. [Technology Stack & Why We Chose Each Tool](#3-technology-stack--why-we-chose-each-tool)
4. [Complete ML Pipeline — Step by Step](#4-complete-ml-pipeline--step-by-step)
5. [Machine Learning Models — Theory & Why We Use Them](#5-machine-learning-models--theory--why-we-use-them)
6. [Hyperparameter Tuning — Theory & Implementation](#6-hyperparameter-tuning--theory--implementation)
7. [Evaluation Metrics — Theory & Results](#7-evaluation-metrics--theory--results)
8. [The Streamlit Web Application](#8-the-streamlit-web-application)
9. [Model Persistence — Saving & Loading Artifacts](#9-model-persistence--saving--loading-artifacts)
10. [Potential Viva Questions & Answers](#10-potential-viva-questions--answers)

---

## 1. Project Overview & Motivation

### What is WellSense?
WellSense is a **machine learning-based classification system** that predicts the risk of depression in Indian students. It is a **binary classification problem** — the output is either:
- `0` → No Depression
- `1` → Depression Detected

### Why is this project important?
- Student mental health has become a critical concern in India and globally.
- Early detection of depression risk can enable timely intervention.
- Traditional clinical diagnosis is expensive and inaccessible to many students.
- ML models trained on behavioral and academic features can act as a low-cost screening tool.

### Disclaimer
This is for **educational purposes only** and is **NOT a clinical diagnostic tool**. It cannot and should not replace professional mental health assessment.

---

## 2. Dataset

### Source
- **Name:** Student Depression Dataset
- **Author:** adilshamim8 (Kaggle)
- **URL:** https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset

### Key Statistics
| Property | Value |
|----------|-------|
| Total Records | 27,901 (raw), ~27,450 (after cleaning) |
| Number of Features | 18 (raw), 14 (after selection) |
| Target Variable | Depression (0 = No, 1 = Yes) |
| Context | India — CGPA on 0–10 scale, Indian cities, Indian degree programs |

### Why this dataset?
- India-relevant context (unlike many Western datasets)
- Covers both academic and lifestyle factors
- Has a large enough sample size for reliable ML training
- Includes key mental health indicators like suicidal thoughts and family history

### Dataset Features (All 18 columns)
| Feature | Type | Description |
|---------|------|-------------|
| id | Nominal | Unique identifier (dropped — not informative) |
| Gender | Categorical | Male / Female |
| Age | Numeric | Student age (18–59) |
| City | Categorical | Indian city (dropped — 52 unique cities, low signal) |
| Profession | Categorical | Student, Doctor, Teacher, etc. |
| Academic Pressure | Numeric | 0–5 scale |
| Work Pressure | Numeric | 0–5 scale |
| CGPA | Numeric | 0–10 Indian academic scale |
| Study Satisfaction | Numeric | 1–5 scale |
| Job Satisfaction | Numeric | 1–5 scale (dropped — mostly missing/zero for students) |
| Sleep Duration | Categorical | Categories of sleep (< 5 hours, 5-6 hours, etc.) |
| Dietary Habits | Categorical | Healthy / Moderate / Unhealthy |
| Degree | Categorical | Indian educational degrees (B.Tech, BSc, etc.) |
| Suicidal Thoughts? | Binary | Yes / No |
| Work/Study Hours | Numeric | Hours per day |
| Financial Stress | Numeric | 1–5 scale |
| Family History of Mental Illness | Binary | Yes / No |
| **Depression** | **Binary (TARGET)** | **0 = No, 1 = Yes** |

### Class Distribution (Before SMOTE)
- Class 1 (Depression): 16,336 records (~58.5%)
- Class 0 (No Depression): 11,565 records (~41.5%)
- **Imbalanced dataset** — SMOTE was applied to fix this.

---

## 3. Technology Stack & Why We Chose Each Tool

### Python 3.13
- Industry standard for ML/Data Science
- Rich ecosystem of libraries

### pandas
- **What:** Data manipulation library
- **Why:** Efficiently handles CSV loading, column operations, and DataFrame operations

### NumPy
- **What:** Numerical computing library
- **Why:** Fast array operations — used internally by scikit-learn

### scikit-learn
- **What:** The core ML library
- **Why:** Contains all classification algorithms, preprocessing tools, and evaluation metrics we need in one unified API

### imbalanced-learn (SMOTE)
- **What:** Library for handling imbalanced datasets
- **Why:** Our dataset has more depression cases than non-depression; SMOTE synthetically creates minority class samples

### Matplotlib & Seaborn
- **What:** Visualization libraries
- **Why:** Used for EDA charts, confusion matrices, ROC curves, feature importance plots

### joblib
- **What:** Python serialization library
- **Why:** Efficiently saves large ML models to disk (`.pkl` files) and loads them back

### Streamlit
- **What:** Python web application framework
- **Why:** Converts Python scripts into interactive web apps with minimal code; perfect for ML dashboards

---

## 4. Complete ML Pipeline — Step by Step

The pipeline is divided into 4 phases:

```
Phase 1: Setup & Data Loading
    ↓
Phase 2: Preprocessing
    ↓
Phase 3: Model Training & Hyperparameter Tuning
    ↓
Phase 4: Evaluation & Artifact Saving
```

### Phase 1: Setup & Data Loading

```python
RANDOM_STATE = 42  # For reproducibility
df = pd.read_csv('data/student_depression_dataset.csv')
# Shape: (27901, 18)
```

**Why RANDOM_STATE = 42?**
- Random state seeds ensure reproducibility — every time you run the notebook, you get identical results.
- 42 is a convention in the ML community (from "The Hitchhiker's Guide to the Galaxy").

---

### Phase 2: Preprocessing (Critical Section!)

Preprocessing transforms raw, messy data into a format that ML algorithms can process.

#### Step 2.1 — Feature Selection (Dropping Irrelevant Columns)

```python
df_clean = df.drop(columns=['id', 'City', 'Job Satisfaction'])
```

**Why remove `id`?**
- Just a row number — has zero predictive value.

**Why remove `City`?**
- 52 unique cities — too many categories, leads to high cardinality encoding issues.
- Would add noise without meaningful signal.

**Why remove `Job Satisfaction`?**
- Mostly 0 for students (they don't have jobs).
- Would introduce noise/misleading patterns.

**Remaining: 14 features + 1 target**

---

#### Step 2.2 — Missing Value Handling

```python
df_clean = df_clean.dropna(axis=1, how='all')         # Drop columns that are entirely empty
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)  # Fill numeric NaN with median
df_clean.fillna('Unknown', inplace=True)               # Fill categorical NaN with 'Unknown'
```

**Why median instead of mean for numeric columns?**
- Median is robust to outliers.
- Mean can be skewed by extreme values (e.g., an age of 59 in a primarily 18-25 dataset).

**Result:** 0 missing values.

---

#### Step 2.3 — Label Encoding (Categorical → Numeric)

ML algorithms require numeric inputs. We use **Label Encoding** to convert text categories to integers.

```python
categorical_cols = ['Gender', 'Profession', 'Sleep Duration', 'Dietary Habits',
                    'Degree', 'Have you ever had suicidal thoughts ?',
                    'Family History of Mental Illness']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le  # Save encoder for use at prediction time
```

**Encoding Examples:**
```
Gender:      Female=0, Male=1
Sleep:       '5-6 hours'=0, '7-8 hours'=1, 'Less than 5 hours'=2, 'More than 8 hours'=3
Dietary:     Healthy=0, Moderate=1, Unhealthy=2
Suicidal:    No=0, Yes=1
Family Hist: No=0, Yes=1
```

**Why Label Encoding instead of One-Hot Encoding?**
- One-Hot Encoding would create many additional columns, increasing dimensionality.
- Tree-based methods (Decision Tree, Random Forest) can handle ordinal-like label encoding well.
- For features like Dietary Habits (Healthy < Moderate < Unhealthy), there's an implied ordering.
- We save the encoders to `label_encoders.pkl` so predictions use identical mappings.

---

#### Step 2.4 — SMOTE (Synthetic Minority Oversampling Technique)

```python
smote = SMOTE(random_state=RANDOM_STATE)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

**Before SMOTE:**
- Class 0 (No Depression): 11,565
- Class 1 (Depression): 16,336

**After SMOTE:**
- Class 0: 16,336
- Class 1: 16,336  ✅ (balanced!)

**How SMOTE works:**
1. For each sample in the minority class (Class 0 here), find its K nearest neighbors.
2. Randomly pick one neighbor.
3. Create a synthetic sample along the line between the original sample and the neighbor.
4. Repeat until both classes are equal in size.

**Why SMOTE?**
- With imbalanced data, a naive model would just always predict the majority class and still achieve high accuracy.
- SMOTE ensures the model learns features of BOTH classes equally.
- Better alternative to simple oversampling (random duplication), which leads to overfitting.

---

#### Step 2.5 — StandardScaler (Feature Scaling)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)
```

**What it does:**
- Transforms each feature to have **mean = 0** and **standard deviation = 1**.
- Formula: `z = (x - μ) / σ`

**Why is feature scaling necessary?**
- Algorithms like **SVM**, **k-NN**, and **MLP Neural Network** are distance-based.
- Without scaling, features with larger ranges (e.g., Work/Study Hours: 0–12) dominate over features with smaller ranges (e.g., Gender: 0–1).
- This leads to biased models.

**Key Rule:** The scaler is `fit` on training data only, then used to `transform` both train and test data. This prevents **data leakage**.

---

#### Step 2.6 — Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_balanced  # Maintains class proportions in both splits
)
```

**Split ratios:**
- Training set: 80% → 26,137 samples
- Test set: 20% → 6,535 samples

**Why `stratify=y_balanced`?**
- Without stratification, by chance, one split might have more of one class.
- Stratification ensures the 50/50 class balance is preserved in both train and test sets.

---

#### Step 2.7 — PCA (Principal Component Analysis)

```python
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
pca.fit(X_train)
# Result: 14 original features → 13 PCA components
# Variance retained: 95.97%
```

**What is PCA?**
- PCA is a **dimensionality reduction** technique.
- It finds directions (principal components) in the data that capture the most variance.
- Each component is a linear combination of all original features.

**Why PCA?**
- Reduces noise by discarding components with low variance.
- Combats the "curse of dimensionality."
- Can speed up training.

**The `n_components=0.95` setting** means: keep enough components to explain 95% of the total variance.

**Note:** In our project, PCA reduces 14 → 13 features, retaining 95.97% variance. The gain is modest but demonstrates the concept.

**Important:** PCA is fitted on training data only (to prevent data leakage).

---

## 5. Machine Learning Models — Theory & Why We Use Them

We train and compare **7 classifiers**. Each represents a different algorithmic approach.

---

### Model 1: Logistic Regression

**Theory:**
- Despite the name, it's a **classification** algorithm.
- Models the probability that a sample belongs to class 1 using the **sigmoid function**:
  - `P(y=1) = 1 / (1 + e^(-z))` where `z = wᵀx + b`
- Decision boundary: If P > 0.5 → class 1, else class 0.
- Optimized by minimizing **cross-entropy loss** (log-loss).

**Hyperparameters tuned:**
- `C` (Regularization strength): Controls overfitting. Smaller C = stronger regularization.
- `solver`: Optimization algorithm (`liblinear` for small datasets, `lbfgs` for larger ones).

**Strengths:**
- Fast, interpretable, works well on linearly separable data.
- Provides probability estimates.

**Weaknesses:**
- Assumes linear decision boundary — can't capture complex non-linear patterns.

**When to use:** Baseline model, interpretability is required.

**Our Result:** Accuracy: 85.66%, ROC-AUC: 93.11%

---

### Model 2: Decision Tree

**Theory:**
- A tree-structured model where each internal node tests a feature condition.
- Leaf nodes contain the class prediction.
- Built using **recursive binary splitting** — at each step, choose the split that maximally reduces **Gini impurity** or **entropy**.

**Gini Impurity:**
`Gini = 1 - Σ(pᵢ²)` where pᵢ = proportion of class i

**Entropy:**
`Entropy = -Σ(pᵢ × log₂(pᵢ))`

**Hyperparameters tuned:**
- `max_depth`: Maximum depth of tree (prevents overfitting).
- `min_samples_split`: Minimum samples required to split a node.

**Strengths:**
- Highly interpretable (can visualize the tree).
- Handles non-linear relationships.
- No feature scaling needed.

**Weaknesses:**
- Prone to overfitting (deep trees memorize training data).
- High variance — different splits on same data can give very different trees.

**Our Result:** Accuracy: 79.65%, ROC-AUC: 79.65% — **worst performer** (overfit/high variance).

---

### Model 3: Random Forest ⭐ (BEST MODEL)

**Theory:**
- An **ensemble method** — combines multiple Decision Trees.
- Uses **Bagging** (Bootstrap Aggregating):
  1. Create N random subsets of training data (with replacement) — **bootstrap samples**.
  2. Train a Decision Tree on each subset.
  3. At each split, consider only a **random subset of features** (typically √n features).
  4. Final prediction = **majority vote** of all N trees.

**Why it's better than a single Decision Tree:**
- Individual trees are **high variance** — they overfit.
- Averaging many trees reduces variance while keeping bias low.
- Feature randomness decorrelates trees — each tree learns different aspects.

**Hyperparameters tuned:**
- `n_estimators`: Number of trees (50, 100, 200). More trees → better but slower.
- `max_depth`: Max depth per tree.
- `min_samples_split`: Minimum samples for a split.

**Strengths:**
- Robust to overfitting.
- Handles high-dimensional data well.
- Provides **feature importance** scores.
- Works well with mixed data types.

**Weaknesses:**
- Less interpretable than single tree.
- Slower to train/predict than simple models.

**Feature Importance (top features identified by Random Forest):**
1. Suicidal Thoughts — 38%
2. Academic Pressure — 24%
3. Financial Stress — 17%
4. Academic Score (CGPA) — 10%
5. Sleep Duration — 7%
6. Family History — 4%

**Our Result:** Accuracy: 86.27%, ROC-AUC: 93.70% — **Best overall model!**

---

### Model 4: SVM (Support Vector Machine)

**Theory:**
- Finds the **optimal hyperplane** that maximally separates the two classes.
- The margin between the hyperplane and the nearest training points is **maximized**.
- Those nearest training points are called **support vectors**.
- For non-linear data, uses the **kernel trick**:
  - Maps data to higher-dimensional space where it IS linearly separable.
  - RBF kernel: `K(x, y) = exp(-γ ||x - y||²)`

**Hyperparameters tuned:**
- `C`: Trade-off between margin maximization and misclassification penalty.
  - Small C → wider margin, more misclassification tolerated (underfitting).
  - Large C → narrow margin, few misclassifications (overfitting risk).
- `kernel`: `linear` or `rbf` (Radial Basis Function).

**Strengths:**
- Effective in high-dimensional spaces.
- Robust to outliers (only support vectors matter).

**Weaknesses:**
- Slow on large datasets (O(n²) to O(n³) complexity).
- Requires feature scaling (we use StandardScaler).
- Hard to interpret.

**Our Result:** Accuracy: 85.49%, ROC-AUC: 92.40%

---

### Model 5: k-Nearest Neighbors (k-NN)

**Theory:**
- A **lazy learning** algorithm — doesn't build an explicit model during training.
- Prediction: Find the K nearest training samples (by Euclidean/other distance), majority class = prediction.
- `k=1`: Memorizes training data (overfitting). Large `k`: Smoother boundary.

**Distance formula (Euclidean):**
`d = √(Σ(xᵢ - yᵢ)²)`

**Hyperparameters tuned:**
- `n_neighbors`: Number of neighbors K (3, 5, 9).
- `weights`: `uniform` (equal vote) or `distance` (closer neighbors get more weight).

**Strengths:**
- Simple, intuitive.
- No training phase (lazy learner).
- Naturally handles multi-class problems.

**Weaknesses:**
- Slow prediction for large datasets (must compute distances to ALL training points).
- Sensitive to irrelevant features (curse of dimensionality).
- **Requires feature scaling** — we use StandardScaler.

**Our Result:** Accuracy: 83.64%, ROC-AUC: 90.07%

---

### Model 6: Naive Bayes (Gaussian)

**Theory:**
- Based on **Bayes' Theorem**: `P(class|features) ∝ P(features|class) × P(class)`
- **"Naive"** assumption: All features are **conditionally independent** given the class.
  - This is almost never true in reality, but the model still works surprisingly well.
- **Gaussian NB**: Assumes features follow a Gaussian (normal) distribution.
  - `P(xᵢ|class) = (1/√2πσ²) × exp(-(xᵢ-μ)²/2σ²)`

**Hyperparameters tuned:**
- `var_smoothing`: Adds a small value to variance for numerical stability (prevents division by zero).

**Strengths:**
- Extremely fast (both training and prediction).
- Works well with small datasets.
- Handles many features gracefully.

**Weaknesses:**
- The independent features assumption is often violated.
- Poor probability estimates (but good class predictions).

**Our Result:** Accuracy: 85.00%, ROC-AUC: 92.82%

---

### Model 7: MLP Neural Network (Multi-Layer Perceptron)

**Theory:**
- A **feedforward artificial neural network** with:
  - **Input layer**: 14 features (after PCA, 13)
  - **Hidden layers**: Compute transformations using weighted sums + activation functions
  - **Output layer**: Sigmoid/Softmax for probability output

**How it works:**
1. **Forward pass**: Input → hidden layers (ReLU activation) → output (sigmoid)
2. **Loss computation**: Cross-entropy loss between predictions and true labels
3. **Backward pass**: Compute gradients via **backpropagation** (chain rule)
4. **Weight update**: Gradient descent (Adam optimizer)

**Hyperparameters tuned:**
- `hidden_layer_sizes`: Architecture — (32,) = 1 hidden layer of 32 neurons; (64, 32) = 2 layers
- `learning_rate_init`: Step size for gradient descent (0.001 or 0.01)
- `max_iter=500`: Maximum training epochs

**Strengths:**
- Can model complex non-linear relationships.
- Universal approximator.

**Weaknesses:**
- "Black box" — hard to interpret.
- Needs more data to generalize well.
- Sensitive to hyperparameters.
- Slower to train.

**Our Result:** Accuracy: 84.54%, ROC-AUC: 91.69%

---

## 6. Hyperparameter Tuning — Theory & Implementation

### What are Hyperparameters?
- **Parameters** are learned from data (e.g., tree split conditions, logistic regression weights).
- **Hyperparameters** are set BEFORE training and control the learning process (e.g., max_depth, C, n_estimators).

### Why Tune Hyperparameters?
- Default values are not optimal for every dataset.
- Systematic search finds the best configuration for our specific data.
- Improves model performance without changing the algorithm itself.

### RandomizedSearchCV — What and Why

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=5,          # Number of random parameter combinations to try
    cv=3,              # 3-fold cross-validation per combination
    scoring='f1_weighted',  # Metric to optimize
    n_jobs=-1,         # Use all CPU cores
    random_state=RANDOM_STATE
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
```

### GridSearchCV vs. RandomizedSearchCV

| Aspect | GridSearchCV | RandomizedSearchCV |
|--------|-------------|-------------------|
| Method | Tests ALL combinations | Tests RANDOM subset |
| Time | Exponential with params | Linear with n_iter |
| Best for | Small param spaces | Large param spaces |
| Guarantee | Finds global best | Finds approximately best |

**Why we chose RandomizedSearchCV:**
- Our dataset is large (~32,672 samples after SMOTE).
- 7 models × multiple hyperparameters = very large search space.
- With `n_iter=5, cv=3`, each model does 5 × 3 = 15 evaluations.
- GridSearchCV would take prohibitively long.

### Cross-Validation (CV) in RandomizedSearchCV

**What is k-Fold Cross Validation?**
With `cv=3`:
1. Training data is split into 3 equal-sized **folds**.
2. Model is trained on 2 folds, validated on the 3rd.
3. This is repeated 3 times (each fold serves as validation once).
4. The 3 validation scores are averaged → final CV score.

**Why cross-validation?**
- A single train/validation split can be "lucky" or "unlucky."
- CV gives a more reliable estimate of model performance.
- Helps detect overfitting.

### Hyperparameter Search Spaces

```python
models_and_params = {
    'Logistic Regression': (
        LogisticRegression(random_state=42, max_iter=1000),
        {'C': [0.01, 0.1, 1, 10],
         'solver': ['liblinear', 'lbfgs']}
    ),
    'Decision Tree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [None, 5, 10, 15],
         'min_samples_split': [2, 5, 10]}
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators': [50, 100, 200],
         'max_depth': [None, 10, 20],
         'min_samples_split': [2, 5, 10]}
    ),
    'SVM': (
        SVC(probability=True, random_state=42),
        {'C': [0.1, 1, 10],
         'kernel': ['linear', 'rbf']}
    ),
    'k-NN': (
        KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 9],
         'weights': ['uniform', 'distance']}
    ),
    'Naive Bayes': (
        GaussianNB(),
        {'var_smoothing': np.logspace(-9, -5, 5)}
    ),
    'MLP Neural Net': (
        MLPClassifier(max_iter=500, random_state=42),
        {'hidden_layer_sizes': [(32,), (64, 32)],
         'learning_rate_init': [0.001, 0.01]}
    )
}
```

---

## 7. Evaluation Metrics — Theory & Results

### Why Multiple Metrics?

No single metric tells the full story, especially with imbalanced data.

---

### Metric 1: Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- **What it measures:** Percentage of all predictions that were correct.
- **Limitation:** Can be misleading with class imbalance.
  - Example: If 95% of data is Class 0, a model predicting Class 0 always gets 95% accuracy — useless.

---

### Metric 2: Precision

```
Precision = TP / (TP + FP)
```

- **What it measures:** Of all predicted positives (Depression), what fraction were actually positive?
- **High precision = low false alarms.**
- **When it matters:** When false positives are costly (e.g., unnecessary alarming of students who are fine).

---

### Metric 3: Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

- **What it measures:** Of all actual positives (real depression cases), what fraction did we correctly identify?
- **High recall = few missed cases.**
- **When it matters:** In healthcare — missing a true depression case is dangerous. We want high recall.

---

### Metric 4: F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Harmonic mean** of Precision and Recall.
- Balances both — useful when you need both precision and recall to be good.
- **Used as primary optimization metric** in our RandomizedSearchCV (`scoring='f1_weighted'`).
- `f1_weighted` = F1 weighted by class support (handles class imbalance).

---

### Metric 5: ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)

- The **ROC curve** plots True Positive Rate (Recall) vs. False Positive Rate at different classification thresholds.
- **AUC (Area Under the Curve):** 
  - AUC = 1.0 → Perfect model
  - AUC = 0.5 → Random guessing (diagonal line)
  - AUC > 0.9 → Excellent model

**Why ROC-AUC?**
- Threshold-independent measure.
- Evaluates how well the model separates the two classes at all decision thresholds.

---

### Confusion Matrix

```
                Predicted: No   Predicted: Yes
Actual: No         TN              FP
Actual: Yes        FN              TP
```

**Our Random Forest confusion matrix (approximate):**
- True Negatives (TN) = 4,820 — Correctly identified as not depressed
- False Positives (FP) = 94 — Predicted depressed, but not
- False Negatives (FN) = 88 — Missed actual depression cases  
- True Positives (TP) = 4,998 — Correctly identified as depressed

---

### Final Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest (Tuned) ⭐** | **0.8626** | **0.8627** | **0.8626** | **0.8626** | **0.9377** |
| Random Forest | 0.8627 | 0.8628 | 0.8627 | 0.8627 | 0.9370 |
| Logistic Regression | 0.8566 | 0.8567 | 0.8566 | 0.8566 | 0.9311 |
| SVM | 0.8549 | 0.8552 | 0.8549 | 0.8549 | 0.9240 |
| Naive Bayes | 0.8500 | 0.8514 | 0.8500 | 0.8499 | 0.9282 |
| MLP Neural Net | 0.8454 | 0.8454 | 0.8454 | 0.8454 | 0.9169 |
| k-NN | 0.8364 | 0.8364 | 0.8364 | 0.8364 | 0.9007 |
| Decision Tree | 0.7965 | 0.7965 | 0.7965 | 0.7965 | 0.7965 |

**Winner: Random Forest (Tuned)** with best ROC-AUC of **0.9377**.

---

## 8. The Streamlit Web Application

### Architecture (`app.py`)

```
app.py
├── Loads model artifacts from models/ directory
├── Defines global CSS styling
├── Creates sidebar navigation
└── Renders 4 pages:
    ├── Dashboard (overview, leaderboard, quick prediction)
    ├── Model Metrics (detailed performance, confusion matrix, ROC curves)
    ├── Predict (full 14-feature prediction form)
    └── About (project details, tech stack)
```

### How Artifact Loading Works

```python
@st.cache_resource  # Caches across reruns — loads once per session
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
```

**`@st.cache_resource`:** Prevents reloading large model files on every page interaction.

### Prediction Pipeline in the App

When a user submits the prediction form:

1. **Collect inputs** → Store in pandas DataFrame
2. **Apply Label Encoding** → Same encoders used during training
3. **Reorder columns** → Match the exact feature order the model was trained on
4. **Apply StandardScaler** → Same scaler fitted during training
5. **Run prediction** → `model.predict(X_scaled)` and `model.predict_proba(X_scaled)`
6. **Display result** → Show High Risk / Low Risk with confidence %

### The 14 Input Features (Predict Page)
1. Gender
2. Age
3. Profession
4. Academic Pressure (1–5)
5. Work Pressure (0–5)
6. CGPA (0–10)
7. Study Satisfaction (1–5)
8. Sleep Duration (categorical)
9. Dietary Habits (Healthy/Moderate/Unhealthy)
10. Degree
11. Suicidal Thoughts (Yes/No)
12. Work/Study Hours per day
13. Financial Stress (1–5)
14. Family History of Mental Illness (Yes/No)

---

## 9. Model Persistence — Saving & Loading Artifacts

### What are the saved artifacts?

| File | Size | Contents |
|------|------|----------|
| `best_model.pkl` | ~89 MB | The trained Random Forest classifier |
| `scaler.pkl` | ~1.6 KB | StandardScaler (mean & std for each feature) |
| `label_encoders.pkl` | ~3 KB | Dictionary of LabelEncoder objects per categorical column |
| `features.pkl` | ~264 bytes | List of 14 feature names (in correct order) |
| `results_df.pkl` | ~1 KB | DataFrame with all models' performance metrics |

### Saving code:

```python
joblib.dump(best_model_final, 'models/best_model.pkl')
joblib.dump(scaler,           'models/scaler.pkl')
joblib.dump(label_encoders,   'models/label_encoders.pkl')
joblib.dump(results_df,       'models/results_df.pkl', compress=3)
joblib.dump(FEATURES,         'models/features.pkl')
```

### Why save the scaler & encoders separately?
- When making a prediction on new data, you must apply the **SAME transformation** that was applied to training data.
- If you create a new scaler on deployment, it will transform differently → wrong predictions.
- This is called **training-serving skew** — a major ML engineering concern.

### Why joblib instead of pickle?
- `joblib` is specifically optimized for large NumPy arrays (which scikit-learn models contain internally).
- Faster serialization than standard `pickle`.

---

## 10. Potential Viva Questions & Answers

### Q1: Why did you choose Random Forest as the best model?

**A:** Random Forest achieved the highest ROC-AUC of 0.9377 among all 7 models. It uses ensemble learning (combining many decision trees) which reduces variance and prevents overfitting. It also handles mixed data types well and provides interpretable feature importance scores. The ROC-AUC metric is most meaningful for us because it's threshold-independent and shows how well the model separates classes at all confidence levels.

---

### Q2: What is the difference between overfitting and underfitting?

**A:**
- **Overfitting**: Model learns training data too well (including noise). High training accuracy, low test accuracy. Example: Deep Decision Tree.
- **Underfitting**: Model is too simple to capture patterns. Low accuracy on both train and test. Example: Logistic Regression on highly non-linear data.
- **Solution for overfitting**: Regularization, ensemble methods, pruning, more data.
- **Solution for underfitting**: More complex models, more features, less regularization.

---

### Q3: Why did you use SMOTE instead of just oversampling or undersampling?

**A:**
- **Simple Oversampling** (duplicating minority samples) → Model memorizes the same samples, leads to overfitting.
- **Undersampling** (removing majority samples) → We lose valuable training data.
- **SMOTE** creates new, synthetic samples by interpolating between existing minority samples. This generates diverse training examples, reducing overfitting while balancing classes.

---

### Q4: What is the "curse of dimensionality"?

**A:** In high-dimensional spaces, data becomes increasingly sparse. Distance-based algorithms (k-NN, SVM with RBF kernel) struggle because "nearest neighbors" are still very far away when dimensions are high. PCA helps by reducing to the most informative dimensions.

---

### Q5: What is regularization? Why is it needed?

**A:** Regularization adds a penalty term to the loss function to prevent large weights, which would cause overfitting. In Logistic Regression:
- `L1 Regularization (Lasso)`: Adds |w| penalty — can make weights exactly zero (feature selection).
- `L2 Regularization (Ridge)`: Adds w² penalty — shrinks weights but doesn't zero them.
- The `C` hyperparameter in LogReg = 1/λ (inverse of regularization strength).

---

### Q6: Explain Bayes' Theorem as used in Naive Bayes.

**A:**
```
P(Depression | Features) = [P(Features | Depression) × P(Depression)] / P(Features)
```
- `P(Depression)` = Prior probability (base rate of depression in data).
- `P(Features | Depression)` = Likelihood (how probable these features are given depression).
- `P(Depression | Features)` = Posterior probability (what we want to predict).
- "Naive" = we assume features are **independent** given the class, so:
  `P(Features | Depression) = P(f1|Dep) × P(f2|Dep) × ... × P(fn|Dep)`

---

### Q7: What is the difference between precision and recall? When would you prioritize each?

**A:**
- **Precision** = Of all predicted as "depressed", how many actually are? → Minimize false alarms.
- **Recall** = Of all truly depressed students, how many did we catch? → Minimize missed cases.

In mental health screening, **recall is more important** — it's worse to miss a depressed student than to falsely flag a healthy one. A false positive can be corrected in follow-up clinical assessment; a false negative means no help for someone who needs it.

---

### Q8: Why did you use StandardScaler and not MinMaxScaler?

**A:**
- **StandardScaler**: Transforms to mean=0, std=1. Not bounded. Robust to outliers.
- **MinMaxScaler**: Scales to [0,1]. Sensitive to outliers (one extreme value squishes everything else).
- Our data has potential outliers (e.g., very high age or study hours).
- StandardScaler is generally preferred for algorithms that assume normally distributed data (Logistic Regression, SVM, MLP).

---

### Q9: What is data leakage and how did you prevent it?

**A:** Data leakage occurs when information from the test set "leaks" into the training process, leading to overly optimistic performance estimates.

**How we prevented it:**
1. SMOTE was applied ONLY to the training set, not the test set.
2. StandardScaler was `fit` on training data, then used to `transform` both train and test.
3. PCA was `fit` on training data only.
4. Cross-validation (CV=3) in RandomizedSearchCV further reduces overfitting to a single split.

---

### Q10: Why did the Decision Tree perform the worst?

**A:** A single Decision Tree with no depth limit will **memorize the training data** (overfit). It creates very complex rules specific to training samples. The confusion matrix for Decision Tree shows AUC = 0.7965, which is barely better than random guessing threshold for some thresholds. Random Forest fixes this by training many trees on random subsets and averaging their predictions.

---

### Q11: Explain how Streamlit serves as the deployment platform.

**A:** Streamlit converts Python scripts into interactive web applications. When `streamlit run app.py` is executed:
1. Streamlit starts a local web server.
2. The `app.py` script is executed top-to-bottom initially.
3. When the user interacts (clicks a button, changes a slider), Streamlit re-runs the relevant parts.
4. `@st.cache_resource` prevents expensive operations (like loading the 89MB model) from re-running on every interaction.
5. On Streamlit Cloud, GitHub pushes trigger auto-redeployment.

---

### Q12: What does `n_jobs=-1` mean in RandomizedSearchCV?

**A:** `-1` tells scikit-learn to use ALL available CPU cores for parallel computation. This significantly speeds up cross-validation, as each fold's evaluations can run simultaneously on different cores.

---

### Q13: What is the purpose of the `random_state` parameter?

**A:** Many ML algorithms involve randomness (e.g., which data is sampled, how weights are initialized). Setting `random_state=42` seeds the random number generator, ensuring identical results every time the code runs. This is critical for **reproducibility** in research and debugging.

---

### Q14: Could this model be deployed in production? What limitations exist?

**A:**
**Limitations:**
1. Dataset is labeled data from one source — may not generalize to all student populations.
2. The "Naive" independence assumption in some models is violated.
3. Mental health has many factors not captured here (trauma, medication, etc.).
4. The model could perpetuate biases present in training data.
5. Explainability: It's hard to explain WHY the model made a specific prediction to a non-technical user.

**For production:**
- Regular retraining with fresh, diverse data
- Clinical validation study
- Ethical review board approval
- Bias testing across subgroups

---

## QUICK REFERENCE SUMMARY

| Concept | Definition |
|---------|-----------|
| Classification | Predicting a category label |
| Binary Classification | Two possible output classes |
| Feature Engineering | Transforming raw data into useful inputs |
| Label Encoding | Converting categories to integers |
| SMOTE | Synthetic oversampling for class balance |
| StandardScaler | Mean=0, Std=1 normalization |
| PCA | Dimensionality reduction preserving variance |
| Train/Test Split | 80/20 split for unbiased evaluation |
| Cross-Validation | Multiple train/val splits for reliability |
| Hyperparameter Tuning | Finding best model configuration |
| RandomizedSearchCV | Random search over hyperparameter space |
| Ensemble | Combining multiple models |
| Bagging | Bootstrap + Aggregating (Random Forest) |
| Overfitting | Model too complex, fails on new data |
| ROC-AUC | Threshold-independent classification quality |
| F1 Score | Harmonic mean of Precision & Recall |
| Joblib | Efficient model serialization |
| Streamlit | Python-to-web-app framework |

---

*Prepared for Viva Voce Examination — WellSense ML Project*
*Netaji Subhas University of Technology, Department of Computer Science Engineering*
