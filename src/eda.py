# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if not exists
os.makedirs('outputs/plots', exist_ok=True)

# Load data
df = pd.read_csv('data/processed/master_features.csv')

print('=== DATASET OVERVIEW ===')
print(df.shape)
print(df.dtypes)
print(df.describe())

# ----- Plot 1: Rating Distribution -----
plt.figure(figsize=(8,5))
sns.countplot(x='Rating', data=df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating (1-5)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/plots/rating_distribution.png', dpi=150)
plt.close()
print('Saved: rating_distribution.png')

# ----- Plot 2: Visit Mode Distribution -----
plt.figure(figsize=(8,5))
mode_map = {1:'Business',2:'Couples',3:'Family',4:'Friends',5:'Solo'}
df['VisitModeName2'] = df['VisitMode'].map(mode_map)
sns.countplot(x='VisitModeName2', data=df,
              order=df['VisitModeName2'].value_counts().index)
plt.title('Visit Mode Distribution')
plt.xlabel('Visit Mode')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/plots/visit_mode_dist.png', dpi=150)
plt.close()
print('Saved: visit_mode_dist.png')

# ----- Plot 3: Average Rating by Attraction Type -----
plt.figure(figsize=(12,6))
type_rating = df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False)
type_rating.plot(kind='bar')
plt.title('Average Rating by Attraction Type')
plt.xlabel('Attraction Type')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/plots/rating_by_type.png', dpi=150)
plt.close()
print('Saved: rating_by_type.png')

# ----- Plot 4: User Distribution by Continent -----
plt.figure(figsize=(8,5))
cont_counts = df.groupby('Continent')['UserId'].nunique().sort_values(ascending=False)
cont_counts.plot(kind='bar')
plt.title('Unique Users by Continent')
plt.xlabel('Continent')
plt.ylabel('Unique Users')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/plots/users_by_continent.png', dpi=150)
plt.close()
print('Saved: users_by_continent.png')

# ----- Plot 5: Monthly Visit Trend -----
plt.figure(figsize=(10,5))
monthly = df.groupby('VisitMonth')['TransactionId'].count()
monthly.plot(kind='line', marker='o')
plt.title('Monthly Visit Trend')
plt.xlabel('Month')
plt.ylabel('Number of Visits')
plt.xticks(range(1,13))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/monthly_trend.png', dpi=150)
plt.close()
print('Saved: monthly_trend.png')

# ----- Plot 6: Heatmap - VisitMode vs Continent -----
plt.figure(figsize=(10,6))
pivot = df.groupby(['Continent','VisitModeName2'])['TransactionId'].count().unstack(fill_value=0)
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Visits by Continent & Visit Mode')
plt.tight_layout()
plt.savefig('outputs/plots/heatmap_continent_mode.png', dpi=150)
plt.close()
print('Saved: heatmap_continent_mode.png')

# ----- Plot 7: Correlation Matrix -----
plt.figure(figsize=(10,8))
num_cols = [
    'Rating','VisitMonth','VisitYear','VisitMode',
    'ContinentId','AttractionTypeId','UserAvgRating','AttrAvgRating'
]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('outputs/plots/correlation_matrix.png', dpi=150)
plt.close()
print('Saved: correlation_matrix.png')

# ----- Top 10 Most Visited Attractions -----
top_attr = df.groupby('Attraction')['TransactionId'].count().nlargest(10)
print('\n=== TOP 10 MOST VISITED ATTRACTIONS ===')
print(top_attr.to_string())

print('\n✅ All EDA plots saved to outputs/plots/')