# 🌏 Tourism Experience Analytics
### Classification · Prediction · Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-337AB7?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-2ecc71?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br/>

> **A full-stack machine learning project that predicts user visit modes, estimates attraction ratings, and delivers personalized travel recommendations — powered by 52,930 real tourism transactions across 5 continents.**

<br/>

[🚀 Live Demo](#-streamlit-app) · [📊 Dataset](#-dataset-overview) · [🧠 Models](#-ml-models) · [⚙️ Setup](#%EF%B8%8F-installation--setup) · [📁 Structure](#-project-structure)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Use Cases](#-business-use-cases)
- [Dataset Overview](#-dataset-overview)
- [Project Structure](#-project-structure)
- [Installation & Setup](#%EF%B8%8F-installation--setup)
- [Running the Pipeline](#-running-the-pipeline)
- [ML Models](#-ml-models)
- [EDA Insights](#-eda-insights)
- [Streamlit App](#-streamlit-app)
- [Model Performance](#-model-performance)
- [Technologies Used](#-technologies-used)
- [Known Issues & Notes](#%EF%B8%8F-known-issues--notes)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

Tourism platforms face a critical challenge: **how to personalize the travel experience at scale.** This project addresses that by building a complete analytics system over real-world tourism data covering users from **Africa, Americas, Asia, Europe, and Australia & Oceania**.

### Three Core Objectives

| # | Task | Goal | Best Model |
|---|------|------|-----------|
| 1 | **Regression** | Predict rating a user will give an attraction | Gradient Boosting (R²=0.7478) |
| 2 | **Classification** | Predict visit mode (Business / Couples / Family / Friends / Solo) | Random Forest (F1=0.4075) |
| 3 | **Recommendation** | Suggest personalized attractions | Collaborative + Content-Based Filtering |

---

## 💼 Business Use Cases

```
┌─────────────────────────────────────────────────────────────────┐
│  🎯 Personalized Recommendations                                │
│     Suggest attractions based on past visits & demographics     │
│                                                                 │
│  📊 Tourism Analytics                                           │
│     Identify hotspots, seasonal trends, and popular regions     │
│                                                                 │
│  👥 Customer Segmentation                                       │
│     Classify travelers → enable targeted promotions             │
│                                                                 │
│  🔁 Customer Retention                                          │
│     Boost loyalty through hyper-personalized suggestions        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Overview

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Transactions | 52,930 |
| Unique Users | 33,530 |
| Unique Attractions | 1,698 |
| Year Range | 2013 – 2022 |
| Rating Scale | 1 – 5 |
| Continents Covered | 5 |
| Attraction Types | 17 |
| Visit Modes | 5 |

### Dataset Files

```
data/raw/Tourism_Dataset/Tourism Dataset/
├── Transaction.xlsx                              ← 52,930 rows | Core dataset
├── User.xlsx                                     ← 33,530 rows | User demographics
├── City.xlsx                                     ← 9,143 rows  | City names & countries
├── Country.xlsx                                  ← 165 rows    | Country lookup
├── Region.xlsx                                   ← 22 rows     | Region lookup
├── Continent.xlsx                                ← 6 rows      | Continent lookup
├── Type.xlsx                                     ← 17 rows     | Attraction type lookup
├── Mode.xlsx                                     ← 6 rows      | Visit mode lookup
├── Item.xlsx                                     ← 30 rows     | Attraction info (basic)
└── Additional_Data_for_Attraction_Sites/
    └── Updated_Item.xlsx                         ← 1,698 rows  | Full attraction catalog
```

> ⚠️ **Windows Note:** After running `Expand-Archive`, the zip extracts to `data\raw\Tourism_Dataset\Tourism Dataset\`. Update `BASE_PATH` in `src/data_cleaning.py` to match this path.

### Schema Relationships

```
Transaction ──── UserId ──────────────────► User
     │                                        │
     ├── AttractionId ──► Updated_Item        ├── ContinentId ──► Continent
     │                        │               ├── RegionId    ──► Region
     │                        ├── AttractionTypeId ──► Type   ├── CountryId  ──► Country
     │                        └── AttractionCityId ──► City   └── CityId     ──► City
     └── VisitMode ──► Mode
```

### Visit Modes

| ID | Mode | Description |
|----|------|-------------|
| 1 | 💼 Business | Work-related travel |
| 2 | 💑 Couples | Romantic travel |
| 3 | 👨‍👩‍👧 Family | Family vacation |
| 4 | 👫 Friends | Group travel with friends |
| 5 | 🧳 Solo | Independent travel |

### Attraction Types

`Ancient Ruins` · `Ballets` · `Beaches` · `Caverns & Caves` · `Flea & Street Markets` · `Historic Sites` · `History Museums` · `National Parks` · `Nature & Wildlife Areas` · `Neighborhoods` · `Points of Interest & Landmarks` · `Religious Sites` · `Spas` · `Speciality Museums` · `Volcanos` · `Water Parks` · `Waterfalls`

---

## 📁 Project Structure

```
Tourism_Experience_Analytics/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 data/
│   ├── raw/
│   │   └── Tourism_Dataset/
│   │       └── Tourism Dataset/
│   │           ├── Transaction.xlsx
│   │           ├── User.xlsx
│   │           ├── City.xlsx
│   │           ├── Item.xlsx
│   │           ├── Type.xlsx
│   │           ├── Mode.xlsx
│   │           ├── Continent.xlsx
│   │           ├── Country.xlsx
│   │           ├── Region.xlsx
│   │           └── Additional_Data_for_Attraction_Sites/
│   │               └── Updated_Item.xlsx
│   └── processed/
│       ├── master_cleaned.csv        ← (52,930 × 21) output of Phase 1
│       └── master_features.csv      ← (52,930 × 32) output of Phase 2
│
├── 📂 src/
│   ├── data_cleaning.py             ← Phase 1: Clean & merge all 9 tables
│   ├── preprocessing.py             ← Phase 2: Feature engineering & encoding
│   ├── eda.py                       ← Phase 3: EDA + 7 visualizations
│   ├── regression_model.py          ← Phase 4a: Predict rating
│   ├── classification_model.py      ← Phase 4b: Predict visit mode
│   └── recommendation.py           ← Phase 4c: Recommend attractions
│
├── 📂 models/
│   ├── regression_model.pkl         ← Gradient Boosting (best, R²=0.7478)
│   ├── classification_model.pkl     ← Random Forest (best, F1=0.4075)
│   ├── encoders.pkl                 ← Label encoders for 4 columns
│   ├── regression_features.pkl      ← 11 feature names list
│   ├── classification_features.pkl  ← 10 feature names list
│   ├── user_item_matrix.pkl         ← (10,618 × 30) user-attraction matrix
│   └── master_df.pkl               ← Cached dataframe for Streamlit
│
├── 📂 outputs/
│   └── plots/
│       ├── rating_distribution.png
│       ├── visit_mode_dist.png
│       ├── rating_by_type.png
│       ├── users_by_continent.png
│       ├── monthly_trend.png
│       ├── heatmap_continent_mode.png
│       └── correlation_matrix.png
│
└── 📂 streamlit_app/
    └── app.py                       ← 4-page interactive dashboard
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- Git installed
- GitHub account

### Step 1 — Create Project Folder (Windows PowerShell)

```powershell
cd E:\

mkdir Tourism_Experience_Analytics
cd Tourism_Experience_Analytics

# PowerShell requires separate mkdir calls (no -p flag for multiple paths)
mkdir data
mkdir data\raw
mkdir data\processed
mkdir notebooks
mkdir src
mkdir models
mkdir outputs
mkdir outputs\plots
mkdir streamlit_app
```

### Step 2 — Create Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm openpyxl streamlit scipy plotly
pip freeze > requirements.txt
```

### Step 4 — Add Dataset

```powershell
# Copy your zip to data\raw\ then extract
Expand-Archive data\raw\Tourism_Dataset.zip -DestinationPath data\raw\Tourism_Dataset
```

After extraction, verify the path:

```powershell
ls data\raw\Tourism_Dataset\"Tourism Dataset"
```

Expected output:
```
Mode    LastWriteTime    Length  Name
----    -------------    ------  ----
d-----  ...                      Additional_Data_for_Attraction_Sites
-a----  ...              212902  City.xlsx
-a----  ...                8825  Continent.xlsx
-a----  ...               13174  Country.xlsx
-a----  ...               10647  Item.xlsx
-a----  ...                8863  Mode.xlsx
-a----  ...                9279  Region.xlsx
-a----  ...             2033119  Transaction.xlsx
-a----  ...                9366  Type.xlsx
-a----  ...              880281  User.xlsx
```

---

## 🚀 Running the Pipeline

Run all scripts from the project root `E:\Tourism_Experience_Analytics\`.

---

### Phase 1 — Data Cleaning

```powershell
python src\data_cleaning.py
```

**What it does:**
- Loads all 9 Excel tables including `Updated_Item.xlsx`
- Removes duplicates, filters out invalid VisitMode (0), bad ratings, and out-of-range months/years
- Fills 4 missing `CityId` values with 0
- Merges all tables into one master DataFrame
- Saves → `data/processed/master_cleaned.csv`

**Actual output:**
```
Transaction original shape: (52930, 7)
Transaction cleaned shape: (52930, 7)
User original shape: (33530, 5)
User cleaned shape: (33530, 5)
Master DataFrame shape: (52930, 21)
Columns: ['TransactionId', 'UserId', 'VisitYear', 'VisitMonth', 'VisitMode',
          'AttractionId', 'Rating', 'ContinentId', 'RegionId', 'CountryId',
          'CityId', 'AttractionCityId', 'AttractionTypeId', 'Attraction',
          'AttractionAddress', 'Continent', 'Region', 'Country',
          'UserCityName', 'AttractionType', 'VisitModeName']

✅ Cleaned data saved to data/processed/master_cleaned.csv
```

---

### Phase 2 — Feature Engineering & Preprocessing

```powershell
python src\preprocessing.py
```

**What it does:**
- Creates `Season` column from VisitMonth
- Aggregates per user: `UserAvgRating`, `UserTotalVisits`, `UserFavMode`
- Aggregates per attraction: `AttrAvgRating`, `AttrTotalVisits`
- Label encodes: Continent, Region, Country, AttractionType, Season
- Saves encoders → `models/encoders.pkl`
- Saves → `data/processed/master_features.csv`

**Actual output:**
```
After feature engineering: (52930, 27)
Encoded Continent: ['Africa' 'America' 'Asia' 'Australia & Oceania' 'Europe']
Encoded Region: ['-' 'Australia' 'Caribbean' 'Central Africa' 'Central America'
 'Central Asia' 'Central Europe' 'East Africa' 'East Asia' 'Eastern Europe'
 'Middle East' 'North Africa' 'Northern America' 'Northern Europe' 'Oceania'
 'South America' 'South Asia' 'South East Asia' 'Southern Africa'
 'Southern Europe' 'West Africa' 'Western Europe']
Encoded AttractionType: ['Ancient Ruins' 'Ballets' 'Beaches' 'Caverns & Caves'
 'Flea & Street Markets' 'Historic Sites' 'History Museums' 'National Parks'
 'Nature & Wildlife Areas' 'Neighborhoods' 'Points of Interest & Landmarks'
 'Religious Sites' 'Spas' 'Speciality Museums' 'Volcanos' 'Water Parks' 'Waterfalls']
Encoded Season: ['Autumn' 'Spring' 'Summer' 'Winter']

✅ Feature engineered data saved to data/processed/master_features.csv
Total features available: 32
```

---

### Phase 3 — Exploratory Data Analysis

```powershell
python src\eda.py
```

**Generates 7 plots saved to `outputs/plots/`:**

| Plot File | Insight |
|-----------|---------|
| `rating_distribution.png` | Ratings heavily skew toward 4–5 |
| `visit_mode_dist.png` | Couples dominate (32%), Family second (23%) |
| `rating_by_type.png` | Waterfalls & Beaches rated highest |
| `users_by_continent.png` | Europe and Asia contribute most users |
| `monthly_trend.png` | Peak visits in October, November, January |
| `heatmap_continent_mode.png` | European users strongly prefer Couples travel |
| `correlation_matrix.png` | `AttrAvgRating` has strongest correlation with `Rating` |

**Actual output:**
```
=== DATASET OVERVIEW ===
(52930, 32)
...
Saved: rating_distribution.png
Saved: visit_mode_dist.png
Saved: rating_by_type.png
Saved: users_by_continent.png
Saved: monthly_trend.png
Saved: heatmap_continent_mode.png
Saved: correlation_matrix.png

=== TOP 10 MOST VISITED ATTRACTIONS ===
Attraction
Sacred Monkey Forest Sanctuary    13198
Waterbom Bali                      6429
Tegalalang Rice Terrace            5815
Uluwatu Temple                     3359
Tanah Lot Temple                   3352
Sanur Beach                        3044
Seminyak Beach                     2914
Kuta Beach - Bali                  2765
Merapi Volcano                     2235
Tegenungan Waterfall               2190

✅ All EDA plots saved to outputs/plots/
```

---

### Phase 4a — Regression Model (Predict Rating)

```powershell
python src\regression_model.py
```

**Features (11):** `ContinentId` · `RegionId` · `CountryId` · `VisitMonth` · `VisitYear` · `VisitMode` · `AttractionTypeId` · `UserAvgRating` · `UserTotalVisits` · `AttrAvgRating` · `AttrTotalVisits`

**Actual output:**
```
Train: (42344, 11), Test: (10586, 11)

=== REGRESSION MODEL COMPARISON ===
Model                           R2     RMSE      MAE
-------------------------------------------------------
Linear Regression           0.7375   0.4972   0.2905
Random Forest               0.6945   0.5364   0.2749
Gradient Boosting           0.7478   0.4874   0.2698
XGBoost                     0.7316   0.5027   0.2733

🏆 Best Model: Gradient Boosting (R2=0.7478)
✅ Best regression model saved to models/regression_model.pkl
```

---

### Phase 4b — Classification Model (Predict Visit Mode)

```powershell
python src\classification_model.py
```

> ⚠️ **Important:** XGBoost requires 0-indexed class labels. VisitMode values (1–5) must be remapped to (0–4) before fitting XGBoost. Target names are restored at the reporting stage.

**Actual output:**
```
Train: (42344, 10), Test: (10586, 10)
Class distribution (train): {0: 498, 1: 17296, 2: 12174, 3: 8756, 4: 3620}

=== CLASSIFICATION MODEL COMPARISON ===
Model                       Accuracy   F1-Macro  Precision     Recall
---------------------------------------------------------------------------
Logistic Regression           0.3984     0.1480     0.1377     0.2025
Random Forest                 0.4971     0.4075     0.4611     0.3843
Gradient Boosting             0.4851     0.2784     0.5534     0.2859
XGBoost                       0.5103     0.3464     0.5549     0.3318

🏆 Best Classifier: Random Forest (F1-Macro=0.4075)

=== Detailed Classification Report (Best Model) ===
              precision    recall  f1-score   support

    Business       0.47      0.23      0.31       125
     Couples       0.53      0.65      0.59      4324
      Family       0.51      0.50      0.51      3043
     Friends       0.39      0.31      0.35      2189
        Solo       0.40      0.22      0.29       905

    accuracy                           0.50     10586
   macro avg       0.46      0.38      0.41     10586
weighted avg       0.49      0.50      0.49     10586

✅ Best classification model saved to models/classification_model.pkl
```

---

### Phase 4c — Recommendation System

```powershell
python src\recommendation.py
```

**Actual output:**
```
User-Item Matrix Shape: (10618, 30)

=== TESTING COLLABORATIVE FILTER ===
Recommendations for User 14
Empty DataFrame
Columns: [Attraction, PredictedRating]
Index: []

=== TESTING CONTENT FILTER (Type 13 Example) ===
       Attraction  AvgRating
   Nusa Dua Beach   4.275665
   Goa Cina Beach   4.092593
Balekambang Beach   4.027778
      Sanur Beach   3.976347
   Seminyak Beach   3.800618

✅ Recommendation models saved to models/
```

> **Note:** Empty collaborative result for User 14 is expected — this user's visited attractions may not overlap with similar users. The Streamlit app automatically falls back to content-based filtering in this case.

---

### Phase 5 — Launch Streamlit App

```powershell
cd streamlit_app
streamlit run app.py
```

**Actual output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.26.82.62:8501
```

Open **http://localhost:8501** in your browser.

---

## 🧠 ML Models

### Feature Engineering Summary

```python
# User-level aggregates (per UserId)
UserAvgRating    = mean(Rating)       per UserId
UserTotalVisits  = count(Transaction) per UserId
UserFavMode      = mode(VisitMode)    per UserId

# Attraction-level aggregates (per AttractionId)
AttrAvgRating    = mean(Rating)       per AttractionId
AttrTotalVisits  = count(Transaction) per AttractionId

# Time feature
Season = { Dec, Jan, Feb → Winter  |  Mar, Apr, May → Spring
           Jun, Jul, Aug → Summer  |  Sep, Oct, Nov → Autumn }
```

### Label Encoded Columns

| Column | Unique Classes |
|--------|---------------|
| `Continent` | 5 (Africa, America, Asia, Australia & Oceania, Europe) |
| `Region` | 22 regions |
| `Country` | 154 countries |
| `AttractionType` | 17 types |
| `Season` | 4 (Autumn, Spring, Summer, Winter) |

### Recommendation Approach

```
Collaborative Filtering
  └── Build User × Attraction rating matrix  →  shape (10,618 × 30)
  └── Compute cosine similarity between user vectors
  └── Find top similar users
  └── Recommend their high-rated, unvisited attractions
  └── Falls back to content-based if no overlap found

Content-Based Filtering
  └── Group attractions by AttractionTypeId
  └── Rank by average rating
  └── Return top-N for selected attraction type
```

---

## 📈 EDA Insights

| Finding | Detail |
|---------|--------|
| **#1 Most Visited** | Sacred Monkey Forest Sanctuary — 13,198 visits |
| **Top Visit Mode** | Couples (32%), Family (23%), Friends (17%) |
| **Peak Travel Months** | October, November, January |
| **Highest Rated Types** | Waterfalls & Beaches |
| **Most Active Regions** | Europe and Asia |
| **Rating Bias** | Majority of ratings are 4 or 5 |
| **Strongest Predictor** | `AttrAvgRating` most correlated with individual `Rating` |
| **Class Imbalance** | Business: 498 samples vs Couples: 17,296 in training set |

---

## 🖥️ Streamlit App

### 🏠 Page 1 — Home & EDA Dashboard
- KPI metrics: total transactions, unique users, attractions, avg rating
- Rating distribution bar chart
- Visit mode pie chart
- Top 10 most visited attractions (led by Sacred Monkey Forest Sanctuary — 13,198)
- Visits by continent bar chart

### 🔮 Page 2 — Predict Visit Mode
- Input: Continent, Region ID, Country ID, Month, Year, Attraction Type, user history stats
- Output: Predicted mode with icon (💼 💑 👨‍👩‍👧 👫 🧳)
- Model: **Random Forest Classifier**

### ⭐ Page 3 — Predict Rating
- Input: 11-feature user + attraction profile
- Output: Predicted rating clipped to 1.00–5.00 with star display
- Model: **Gradient Boosting Regressor**

### 💡 Page 4 — Get Recommendations
- **Tab 1 — Collaborative:** Enter User ID → top-N personalized attractions from similar users
- **Tab 2 — Content-Based:** Select attraction type → top-N highest-rated attractions of that type

---

## 📉 Model Performance

### Regression Results

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.7375 | 0.4972 | 0.2905 |
| Random Forest | 0.6945 | 0.5364 | 0.2749 |
| **Gradient Boosting** ✅ | **0.7478** | **0.4874** | **0.2698** |
| XGBoost | 0.7316 | 0.5027 | 0.2733 |

> **Best:** Gradient Boosting with R²=0.7478 — explains 74.78% of variance in user ratings.

### Classification Results

| Model | Accuracy | F1-Macro | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 0.3984 | 0.1480 | 0.1377 | 0.2025 |
| **Random Forest** ✅ | 0.4971 | **0.4075** | 0.4611 | 0.3843 |
| Gradient Boosting | 0.4851 | 0.2784 | 0.5534 | 0.2859 |
| XGBoost | 0.5103 | 0.3464 | 0.5549 | 0.3318 |

> **Best by F1-Macro:** Random Forest (0.4075). XGBoost achieves higher accuracy but lower macro F1 due to severe class imbalance (Business: 498 vs Couples: 17,296).

### Classification Report — Random Forest (Best F1)

```
              precision    recall  f1-score   support

    Business       0.47      0.23      0.31       125
     Couples       0.53      0.65      0.59      4324
      Family       0.51      0.50      0.51      3043
     Friends       0.39      0.31      0.35      2189
        Solo       0.40      0.22      0.29       905

    accuracy                           0.50     10586
   macro avg       0.46      0.38      0.41     10586
weighted avg       0.49      0.50      0.49     10586
```

### Recommendation — User-Item Matrix

| Metric | Value |
|--------|-------|
| Matrix Shape | 10,618 users × 30 attractions |
| Similarity Method | Cosine Similarity |
| Cold Start Handling | Falls back to Content-Based Filtering |

---

## 🛠️ Technologies Used

| Category | Tools & Versions |
|----------|-----------------|
| **Language** | Python 3.12 |
| **Data Processing** | Pandas 2.3.3, NumPy 2.4.2 |
| **Visualization** | Matplotlib 3.10.8, Seaborn 0.13.2, Plotly 6.5.2 |
| **Machine Learning** | Scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0 |
| **Recommendation** | SciPy 1.17.1 (cosine similarity) |
| **Web App** | Streamlit 1.54.0 |
| **File Handling** | OpenPyXL 3.1.5 |
| **Version Control** | Git, GitHub |

---

## 📦 Requirements

```txt
pandas==2.3.3
numpy==2.4.2
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
xgboost==3.2.0
lightgbm==4.6.0
openpyxl==3.1.5
streamlit==1.54.0
scipy==1.17.1
plotly==6.5.2
```

Install all:

```powershell
pip install -r requirements.txt
```

---

## 🔄 Git Workflow

### Staging & Committing

```powershell
# Stage source code and app (exclude data/, models/, outputs/)
git add src
git add streamlit_app
git add requirements.txt
git add README.md
git add .gitignore

git status
```

**Expected git status:**
```
On branch main
Changes to be committed:
  new file:   requirements.txt
  new file:   src/classification_model.py
  new file:   src/data_cleaning.py
  new file:   src/eda.py
  new file:   src/preprocessing.py
  new file:   src/recommendation.py
  new file:   src/regression_model.py
  new file:   streamlit_app/app.py
```

```powershell
git commit -m "feat: complete tourism analytics ML app with streamlit dashboard"
git push
```

**Actual push output:**
```
[main e9833c5] feat: complete tourism analytics ML app with streamlit dashboard
 8 files changed, 811 insertions(+)
 create mode 100644 requirements.txt
 create mode 100644 src/classification_model.py
 create mode 100644 src/data_cleaning.py
 create mode 100644 src/eda.py
 create mode 100644 src/preprocessing.py
 create mode 100644 src/recommendation.py
 create mode 100644 src/regression_model.py
 create mode 100644 streamlit_app/app.py

To https://github.com/Lokesh-7368/Tourism-Experience-Analytics.git
   80b7cef..e9833c5  main -> main
```

### Recommended `.gitignore`

```gitignore
__pycache__/
*.py[cod]
*.pkl
*.joblib
.env
.DS_Store
venv/
data/raw/
data/processed/
models/
outputs/
```

### Commit Message Convention

```
feat:     New feature or script
fix:      Bug fix
data:     Dataset changes or new processed files
model:    Model training, tuning, saving
docs:     README or documentation update
refactor: Code cleanup
```

---

## ⚠️ Known Issues & Notes

| Issue | Detail | Fix |
|-------|--------|-----|
| PowerShell `mkdir -p` | Doesn't accept multiple paths | Use separate `mkdir` calls per folder |
| PowerShell `unzip` | Not available by default | Use `Expand-Archive` instead |
| XGBoost class labels error | Expects 0-indexed labels — VisitMode is 1–5 | Subtract 1 from VisitMode before fitting XGBoost |
| Logistic Regression convergence | `lbfgs` hits 500 iteration limit | Increase `max_iter` or apply `StandardScaler` |
| Collaborative filter empty result | User's visited attractions don't overlap with similar users | App auto-falls back to content-based filtering |
| `SettingWithCopyWarning` in clean_city | `.str.strip()` on a DataFrame slice | Use `.loc[]` indexer to fix |

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "feat: add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a **Pull Request**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Lokesh**
- GitHub: [@Lokesh-7368](https://github.com/Lokesh-7368)

---

<div align="center">

⭐ **Star this repo if you found it helpful!** ⭐

Made with ❤️ for Tourism Analytics

</div>
