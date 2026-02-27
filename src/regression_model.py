# src/regression_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pickle
import os

# Load data
df = pd.read_csv('data/processed/master_features.csv')

FEATURE_COLS = [
    'ContinentId', 'RegionId', 'CountryId',
    'VisitMonth', 'VisitYear',
    'VisitMode', 'AttractionTypeId',
    'UserAvgRating', 'UserTotalVisits',
    'AttrAvgRating', 'AttrTotalVisits'
]

X = df[FEATURE_COLS].fillna(0)
y = df['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Train: {X_train.shape}, Test: {X_test.shape}')

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=100, random_state=42, verbosity=0
    )
}

print('\n=== REGRESSION MODEL COMPARISON ===')
print(f'{"Model":<25} {"R2":>8} {"RMSE":>8} {"MAE":>8}')
print('-'*55)

best_model = None
best_r2 = -999

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f'{name:<25} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}')

    if r2 > best_r2:
        best_r2 = r2
        best_model = (name, model)

print(f'\n🏆 Best Model: {best_model[0]} (R2={best_r2:.4f})')

# Save best model
os.makedirs('models', exist_ok=True)

with open('models/regression_model.pkl', 'wb') as f:
    pickle.dump(best_model[1], f)

with open('models/regression_features.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)

print('✅ Best regression model saved to models/regression_model.pkl')