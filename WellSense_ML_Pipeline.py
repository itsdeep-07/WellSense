# !pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
RANDOM_STATE = 42

os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)
print('✅ All libraries imported!')

df = pd.read_csv('data/student_depression_dataset.csv')
print(f'✅ Dataset loaded! Shape: {df.shape}')
df.head()

print('Columns:', df.columns.tolist())
print('\nTarget distribution:')
print(df['Depression'].value_counts())
df.describe(include='all')

# Missing values
missing = df.isnull().sum()
missing_df = pd.DataFrame({'Missing': missing, 'Pct': (missing/len(df))*100})
missing_df = missing_df[missing_df['Missing'] > 0]
print('✅ No missing values!' if missing_df.empty else missing_df)

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df['Depression'].value_counts()
axes[0].bar(['No Depression (0)', 'Depression (1)'], counts.values, color=['#2ecc71','#e74c3c'], width=0.5)
axes[0].set_title('Depression Distribution', fontsize=13, fontweight='bold')
for i, v in enumerate(counts.values): axes[0].text(i, v+5, str(v), ha='center', fontweight='bold')
axes[1].pie(counts.values, labels=['No Depression','Depression'], colors=['#2ecc71','#e74c3c'], autopct='%1.1f%%')
axes[1].set_title('Depression Split', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Demographics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
df['Gender'].value_counts().plot(kind='pie', ax=axes[0], autopct='%1.1f%%', colors=['#74b9ff','#fd79a8'])
axes[0].set_title('Gender Distribution', fontsize=13, fontweight='bold'); axes[0].set_ylabel('')
sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=axes[1], color='#6c5ce7')
axes[1].set_title('Age Distribution', fontsize=13, fontweight='bold')
df['Profession'].value_counts().head(6).plot(kind='barh', ax=axes[2], color='#00b894')
axes[2].set_title('Top Professions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/demographics.png', dpi=150, bbox_inches='tight')
plt.show()

# Academic features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['CGPA'].dropna(), bins=20, kde=True, ax=axes[0], color='#e17055')
axes[0].set_title('CGPA Distribution (0–10)', fontsize=13, fontweight='bold')
df['Academic Pressure'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='#fdcb6e')
axes[1].set_title('Academic Pressure Levels', fontsize=13, fontweight='bold'); axes[1].tick_params(rotation=0)
df['Sleep Duration'].value_counts().plot(kind='bar', ax=axes[2], color='#a29bfe')
axes[2].set_title('Sleep Duration', fontsize=13, fontweight='bold'); axes[2].tick_params(rotation=30)
plt.tight_layout()
plt.savefig('assets/academic_features.png', dpi=150, bbox_inches='tight')
plt.show()

# Handle missing values represented as '?'
df.replace('?', np.nan, inplace=True)
# Convert numeric columns to float
num_cols = ['Age','Academic Pressure','Work Pressure','CGPA','Study Satisfaction',
            'Job Satisfaction','Work/Study Hours','Financial Stress','Depression']
df[num_cols] = df[num_cols].astype(float)


# Depression by key features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
pd.crosstab(df['Gender'], df['Depression']).plot(kind='bar', ax=axes[0,0], color=['#2ecc71','#e74c3c'])
axes[0,0].set_title('Depression by Gender', fontsize=12, fontweight='bold'); axes[0,0].tick_params(rotation=0)
pd.crosstab(df['Academic Pressure'], df['Depression']).plot(kind='bar', ax=axes[0,1], color=['#3498db','#e74c3c'])
axes[0,1].set_title('Depression by Academic Pressure', fontsize=12, fontweight='bold'); axes[0,1].tick_params(rotation=0)
df.boxplot(column='CGPA', by='Depression', ax=axes[1,0])
axes[1,0].set_title('CGPA vs Depression', fontsize=12, fontweight='bold'); axes[1,0].set_xlabel('Depression (0=No, 1=Yes)')
pd.crosstab(df['Financial Stress'], df['Depression']).plot(kind='bar', ax=axes[1,1], color=['#00b894','#e74c3c'])
axes[1,1].set_title('Depression by Financial Stress', fontsize=12, fontweight='bold'); axes[1,1].tick_params(rotation=0)
plt.suptitle('Depression Analysis by Key Features', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../assets/depression_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# Correlation heatmap
num_cols = ['Age','Academic Pressure','Work Pressure','CGPA','Study Satisfaction',
            'Job Satisfaction','Work/Study Hours','Financial Stress','Depression']
plt.figure(figsize=(11, 8))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5, mask=mask, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()



# Drop irrelevant columns
df_clean = df.drop(columns=['id', 'City', 'Job Satisfaction'])
print(f'Remaining columns: {df_clean.columns.tolist()}')

# Handle missing values
df_clean = df_clean.dropna(axis=1, how='all')
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
df_clean.fillna('Unknown', inplace=True)
print('✅ Missing values handled! Remaining:', df_clean.isnull().sum().sum())

# Label Encoding
categorical_cols = df_clean.select_dtypes(include='object').columns.tolist()
print(f'Encoding: {categorical_cols}')
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
    print(f'  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}')
print('✅ Encoding complete!')

# Feature / Target split
TARGET   = 'Depression'
FEATURES = [col for col in df_clean.columns if col != TARGET]
X = df_clean[FEATURES]
y = df_clean[TARGET]
print(f'Features ({len(FEATURES)}): {FEATURES}')
print(f'Class balance: {dict(y.value_counts())}')

# SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f'Before SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}')
print(f'After  SMOTE: {dict(zip(*np.unique(y_balanced, return_counts=True)))}')

# StandardScaler + Train/Test Split
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=RANDOM_STATE, stratify=y_balanced
)
print(f'Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# PCA
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
pca.fit(X_train)
print(f'Original features: {X_train.shape[1]} → PCA components: {pca.n_components_}')
print(f'Variance retained: {sum(pca.explained_variance_ratio_):.2%}')
plt.figure(figsize=(9, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='#6c5ce7', lw=2)
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
plt.xlabel('Components'); plt.ylabel('Cumulative Variance')
plt.title('PCA — Explained Variance', fontsize=13, fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('../assets/pca_variance.png', dpi=150, bbox_inches='tight')
plt.show()

models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Decision Tree'      : DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'SVM'                : SVC(probability=True, random_state=RANDOM_STATE),
    'k-NN'               : KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes'        : GaussianNB(),
    'MLP Neural Net'     : MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=RANDOM_STATE)
}
print(f'✅ {len(models)} models ready!')

results = {}; trained_models = {}
for name, model in models.items():
    print(f'Training {name}...', end=' ')
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else None
    results[name] = {
        'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'Recall'   : round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'F1 Score' : round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else 'N/A'
    }
    trained_models[name] = model
    print(f"Acc={results[name]['Accuracy']} ✅")
print('\n🎉 All models trained!')

results_df = pd.DataFrame(results).T.sort_values('F1 Score', ascending=False)
results_df.index.name = 'Model'
results_df

plot_df = results_df[['Accuracy','Precision','Recall','F1 Score']].astype(float)
ax = plot_df.plot(kind='bar', figsize=(14,6), width=0.75, colormap='Set2', edgecolor='white')
ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
ax.set_ylim(0, 1.1); ax.tick_params(rotation=30); ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../assets/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(20, 10)); axes = axes.flatten()
for idx, (name, model) in enumerate(trained_models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Dep','Dep'], yticklabels=['No Dep','Dep'])
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted'); axes[idx].set_ylabel('Actual')
fig.delaxes(axes[-1])
plt.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../assets/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(11, 7))
colors_roc = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22']
for (name, model), color in zip(trained_models.items(), colors_roc):
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test,y_proba):.3f})', color=color, lw=2)
plt.plot([0,1],[0,1],'k--', label='Random', lw=1)
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC-AUC Curves', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9); plt.tight_layout()
plt.savefig('../assets/roc_auc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

best_name = results_df.index[0]
print(f'🏆 Best Model: {best_name}')
print(classification_report(y_test, trained_models[best_name].predict(X_test),
      target_names=['No Depression','Depression']))

rf = trained_models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
plt.figure(figsize=(9,6))
importances.plot(kind='barh', color='#6c5ce7')
plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../assets/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print('5-Fold Cross Validation:\n')
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_balanced, cv=kfold, scoring='f1_weighted')
    print(f'{name:25s} → F1: {scores.mean():.4f} ± {scores.std():.4f}')

param_grid = {
    'n_estimators'     : [50, 100, 200],
    'max_depth'        : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print(f'\n✅ Best Params : {grid_search.best_params_}')
print(f'✅ Best F1     : {grid_search.best_score_:.4f}')

best_model_final = grid_search.best_estimator_
y_pred_tuned     = best_model_final.predict(X_test)
print('🏆 Tuned Random Forest:')
print(f'  Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}')
print(f'  F1 Score : {f1_score(y_test, y_pred_tuned, average="weighted"):.4f}')
print(f'  ROC-AUC  : {roc_auc_score(y_test, best_model_final.predict_proba(X_test)[:,1]):.4f}')

joblib.dump(best_model_final, 'models/best_model.pkl')
joblib.dump(scaler,           'models/scaler.pkl')
joblib.dump(label_encoders,   'models/label_encoders.pkl')
joblib.dump(results_df,       'models/results_df.pkl')
joblib.dump(FEATURES,         'models/features.pkl')
print('✅ All artifacts saved!')
print('🚀 Now push to GitHub and Streamlit Cloud will auto-redeploy!')

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

results_df = joblib.load('models/results_df.pkl')

y_pred  = best_model_final.predict(X_test)
y_proba = best_model_final.predict_proba(X_test)[:,1]

results_df.loc['Random Forest (Tuned)'] = {
    'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
    'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
    'Recall'   : round(recall_score(y_test, y_pred, average='weighted'), 4),
    'F1 Score' : round(f1_score(y_test, y_pred, average='weighted'), 4),
    'ROC-AUC'  : round(roc_auc_score(y_test, y_proba), 4),
}

results_df = results_df.sort_values('F1 Score', ascending=False)
joblib.dump(results_df, 'models/results_df.pkl', compress=3)

print("✅ Saved! Index now:")
print(results_df.index.tolist())
print(results_df)


# import joblib
# from pathlib import Path
# BASE_DIR = Path('.')
# m = joblib.load(BASE_DIR / 'models' / 'best_model.pkl')
# f = joblib.load(BASE_DIR / 'models' / 'features.pkl')
# print('model type:', type(m).__name__)
# print('features type:', type(f).__name__)
# print('features:', f)

# lines = open('app.py', encoding='utf-8').readlines()
# for i, l in enumerate(lines, 1):
#     if 'return model' in l or 'load_artifacts()' in l or 'load_artifacts.clear' in l:
#         print(f'Line {i}: {l.rstrip()}')


import joblib
from pathlib import Path
BASE_DIR = Path('.')
m = joblib.load(BASE_DIR / 'models' / 'best_model.pkl')
s = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
e = joblib.load(BASE_DIR / 'models' / 'label_encoders.pkl')
f = joblib.load(BASE_DIR / 'models' / 'features.pkl')
r = joblib.load(BASE_DIR / 'models' / 'results_df.pkl')
print('model:', type(m).__name__)
print('scaler:', type(s).__name__)
print('encoders:', type(e).__name__)
print('features:', type(f).__name__, len(f))
print('results:', type(r).__name__, r.shape)
print('ALL GOOD - no errors!')

