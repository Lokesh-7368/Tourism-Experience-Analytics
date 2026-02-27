# src/recommendation.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load feature dataset
df = pd.read_csv('data/processed/master_features.csv')


# ==============================
# COLLABORATIVE FILTERING
# ==============================

def build_user_item_matrix(df):
    """Build user-item rating matrix"""

    # Keep users with at least 2 visits
    user_counts = df['UserId'].value_counts()
    active_users = user_counts[user_counts >= 2].index
    df_active = df[df['UserId'].isin(active_users)]

    matrix = df_active.pivot_table(
        index='UserId',
        columns='AttractionId',
        values='Rating',
        aggfunc='mean'
    ).fillna(0)

    print(f'User-Item Matrix Shape: {matrix.shape}')
    return matrix


def get_similar_users(user_id, matrix, top_n=10):
    """Find similar users using cosine similarity"""

    if user_id not in matrix.index:
        return []

    user_vec = matrix.loc[user_id].values.reshape(1, -1)
    similarity = cosine_similarity(user_vec, matrix.values)[0]

    sim_series = pd.Series(similarity, index=matrix.index)
    sim_series = sim_series.drop(user_id, errors='ignore')

    return sim_series.nlargest(top_n).index.tolist()


def collaborative_recommend(user_id, matrix, df_full, top_n=5):
    """Recommend attractions based on similar users"""

    similar_users = get_similar_users(user_id, matrix, top_n=20)

    # Fallback to content-based if no similar users
    if not similar_users:
        default_type = df_full['AttractionTypeId'].mode()[0]
        return content_recommend(df_full, default_type, top_n)

    # Attractions already visited
    visited = set(
        df_full[df_full['UserId'] == user_id]['AttractionId'].unique()
    )

    # Ratings from similar users
    sim_data = df_full[df_full['UserId'].isin(similar_users)]
    sim_data = sim_data[~sim_data['AttractionId'].isin(visited)]

    recs = (
        sim_data.groupby(['AttractionId', 'Attraction'])['Rating']
        .mean()
        .reset_index()
        .sort_values('Rating', ascending=False)
        .head(top_n)
    )

    return recs[['Attraction', 'Rating']].rename(
        columns={'Rating': 'PredictedRating'}
    )


# ==============================
# CONTENT-BASED FILTERING
# ==============================

def content_recommend(df, attraction_type_id, top_n=5):
    """Recommend based on attraction type"""

    recs = (
        df[df['AttractionTypeId'] == attraction_type_id]
        .groupby(['AttractionId', 'Attraction'])['Rating']
        .mean()
        .reset_index()
        .sort_values('Rating', ascending=False)
        .head(top_n)
    )

    return recs[['Attraction', 'Rating']].rename(
        columns={'Rating': 'AvgRating'}
    )


# ==============================
# MAIN
# ==============================

if __name__ == '__main__':

    matrix = build_user_item_matrix(df)

    os.makedirs('models', exist_ok=True)

    # Save for Streamlit
    with open('models/user_item_matrix.pkl', 'wb') as f:
        pickle.dump(matrix, f)

    with open('models/master_df.pkl', 'wb') as f:
        pickle.dump(df, f)

    print('\n=== TESTING COLLABORATIVE FILTER ===')

    test_user = matrix.index[0]
    print(f'Recommendations for User {test_user}')

    recs = collaborative_recommend(test_user, matrix, df)
    print(recs.to_string(index=False))

    print('\n=== TESTING CONTENT FILTER (Type 13 Example) ===')
    recs2 = content_recommend(df, attraction_type_id=13, top_n=5)
    print(recs2.to_string(index=False))

    print('\n✅ Recommendation models saved to models/')