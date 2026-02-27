# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os


def load_master():
    return pd.read_csv('data/processed/master_cleaned.csv')


def feature_engineering(df):
    """Create new features from existing columns"""

    # Season from month
    df['Season'] = df['VisitMonth'].apply(lambda m:
        'Winter' if m in [12, 1, 2] else
        'Spring' if m in [3, 4, 5] else
        'Summer' if m in [6, 7, 8] else
        'Autumn'
    )

    # User-level aggregate features
    user_stats = df.groupby('UserId').agg(
        UserAvgRating=('Rating', 'mean'),
        UserTotalVisits=('TransactionId', 'count'),
        UserFavMode=('VisitMode', lambda x: x.mode()[0])
    ).reset_index()

    df = df.merge(user_stats, on='UserId', how='left')

    # Attraction-level aggregate features
    attr_stats = df.groupby('AttractionId').agg(
        AttrAvgRating=('Rating', 'mean'),
        AttrTotalVisits=('TransactionId', 'count')
    ).reset_index()

    df = df.merge(attr_stats, on='AttractionId', how='left')

    print(f'After feature engineering: {df.shape}')
    return df


def encode_features(df):
    """Label encode categorical columns"""

    cat_cols = ['Continent', 'Region', 'Country', 'AttractionType', 'Season']
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
        encoders[col] = le
        print(f'Encoded {col}: {le.classes_}')

    # Save encoders
    os.makedirs('models', exist_ok=True)
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    return df, encoders


def prepare_regression_data(df):
    feature_cols = [
        'ContinentId', 'RegionId', 'CountryId',
        'VisitMonth', 'VisitYear', 'VisitMode',
        'AttractionTypeId',
        'UserAvgRating', 'UserTotalVisits',
        'AttrAvgRating', 'AttrTotalVisits'
    ]

    X = df[feature_cols].fillna(0)
    y = df['Rating']

    return X, y


def prepare_classification_data(df):
    feature_cols = [
        'ContinentId', 'RegionId', 'CountryId',
        'VisitMonth', 'VisitYear',
        'AttractionTypeId',
        'UserAvgRating', 'UserTotalVisits',
        'AttrAvgRating', 'AttrTotalVisits'
    ]

    X = df[feature_cols].fillna(0)
    y = df['VisitMode']

    return X, y


if __name__ == '__main__':
    df = load_master()
    df = feature_engineering(df)
    df, encoders = encode_features(df)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/master_features.csv', index=False)

    print('\n✅ Feature engineered data saved to data/processed/master_features.csv')
    print(f'Total features available: {df.shape[1]}')