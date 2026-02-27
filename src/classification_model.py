# src/classification_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report
)
import xgboost as xgb
import pickle
import os

# Load data
df = pd.read_csv('data/processed/master_features.csv')

FEATURE_COLS = [
    'ContinentId', 'RegionId', 'CountryId',
    'VisitMonth', 'VisitYear',
    'AttractionTypeId',
    'UserAvgRating', 'UserTotalVisits',
    'AttrAvgRating', 'AttrTotalVisits'
]

X = df[FEATURE_COLS].fillna(0)
y = df['VisitMode'] - 1  # 1=Business,2=Couples,3=Family,4=Friends,5=Solo

# Train-test split (stratified for balanced classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'Train: {X_train.shape}, Test: {X_test.shape}')
print(f'Class distribution (train): {dict(y_train.value_counts().sort_index())}')

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        verbosity=0,
        eval_metric='mlogloss'
    )
}

print('\n=== CLASSIFICATION MODEL COMPARISON ===')
print(f'{"Model":<25} {"Accuracy":>10} {"F1-Macro":>10} {"Precision":>10} {"Recall":>10}')
print('-'*75)

best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f'{name:<25} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}')

    if f1 > best_f1:
        best_f1 = f1
        best_model = (name, model)

print(f'\n🏆 Best Classifier: {best_model[0]} (F1-Macro={best_f1:.4f})')

# Detailed report
y_pred_best = best_model[1].predict(X_test)

mode_names = {
    0:'Business',
    1:'Couples',
    2:'Family',
    3:'Friends',
    4:'Solo'
}

target_names = [mode_names[i] for i in sorted(y.unique())]

print('\n=== Detailed Classification Report (Best Model) ===')
print(classification_report(y_test, y_pred_best, target_names=target_names))

# Save best model
os.makedirs('models', exist_ok=True)

with open('models/classification_model.pkl', 'wb') as f:
    pickle.dump(best_model[1], f)

with open('models/classification_features.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)

print('✅ Best classification model saved to models/classification_model.pkl')