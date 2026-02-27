# streamlit_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append("..")

st.set_page_config(
    page_title="Tourism Analytics",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    with open("../models/regression_model.pkl", "rb") as f:
        reg_model = pickle.load(f)

    with open("../models/classification_model.pkl", "rb") as f:
        clf_model = pickle.load(f)

    with open("../models/user_item_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)

    with open("../models/master_df.pkl", "rb") as f:
        df = pickle.load(f)

    return reg_model, clf_model, matrix, df


reg_model, clf_model, matrix, df = load_models()

MODE_MAP = {1:'Business',2:'Couples',3:'Family',4:'Friends',5:'Solo'}
TYPE_MAP = dict(df[['AttractionTypeId','AttractionType']].drop_duplicates().values)
CONT_MAP = dict(df[['ContinentId','Continent']].drop_duplicates().values)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("🌍 Tourism Analytics")
page = st.sidebar.radio("Navigate", [
    "🏠 Home & EDA",
    "🔎 Predict Visit Mode",
    "⭐ Predict Rating",
    "🎯 Get Recommendations"
])

# ==============================
# PAGE 1: HOME
# ==============================
if page == "🏠 Home & EDA":

    st.title("🌍 Tourism Experience Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Unique Users", f"{df['UserId'].nunique():,}")
    col3.metric("Unique Attractions", f"{df['AttractionId'].nunique():,}")
    col4.metric("Average Rating", f"{df['Rating'].mean():.2f}")

    st.subheader("Rating Distribution")

    fig, ax = plt.subplots()
    df['Rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# ==============================
# PAGE 2: CLASSIFICATION
# ==============================
elif page == "🔎 Predict Visit Mode":

    st.title("🔎 Predict Visit Mode")

    continent_id = st.selectbox("Continent", list(CONT_MAP.keys()),
                                 format_func=lambda x: CONT_MAP[x])

    region_id = st.number_input("Region ID", 1, 21, 5)
    country_id = st.number_input("Country ID", 1, 165, 50)
    month = st.slider("Visit Month", 1, 12, 6)
    year = st.slider("Visit Year", 2018, 2024, 2022)
    attr_type = st.selectbox("Attraction Type", list(TYPE_MAP.keys()),
                             format_func=lambda x: TYPE_MAP[x])
    user_avg = st.slider("Your Avg Past Rating", 1.0, 5.0, 3.5)
    user_visits = st.number_input("Your Total Past Visits", 1, 100, 5)
    attr_avg = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0)
    attr_visits = st.number_input("Attraction Total Visits", 1, 5000, 200)

    if st.button("Predict Visit Mode"):
        features = np.array([[continent_id, region_id, country_id,
                              month, year, attr_type,
                              user_avg, user_visits,
                              attr_avg, attr_visits]])

        pred = clf_model.predict(features)[0]
        st.success(f"Predicted Visit Mode: {MODE_MAP[int(pred)]}")

# ==============================
# PAGE 3: REGRESSION
# ==============================
elif page == "⭐ Predict Rating":

    st.title("⭐ Predict Rating")

    continent_id = st.selectbox("Continent", list(CONT_MAP.keys()),
                                 format_func=lambda x: CONT_MAP[x])

    region_id = st.number_input("Region ID", 1, 21, 5)
    country_id = st.number_input("Country ID", 1, 165, 50)
    month = st.slider("Visit Month", 1, 12, 6)
    year = st.slider("Visit Year", 2018, 2024, 2022)
    visit_mode = st.selectbox("Visit Mode", list(MODE_MAP.keys()),
                               format_func=lambda x: MODE_MAP[x])
    attr_type = st.selectbox("Attraction Type", list(TYPE_MAP.keys()),
                             format_func=lambda x: TYPE_MAP[x])
    user_avg = st.slider("Your Avg Past Rating", 1.0, 5.0, 3.5)
    user_visits = st.number_input("Your Past Visits", 1, 100, 5)
    attr_avg = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0)
    attr_visits = st.number_input("Attraction Total Visits", 1, 5000, 200)

    if st.button("Predict Rating"):
        features = np.array([[continent_id, region_id, country_id,
                              month, year, visit_mode,
                              attr_type, user_avg, user_visits,
                              attr_avg, attr_visits]])

        pred = reg_model.predict(features)[0]
        pred = np.clip(pred, 1, 5)

        st.success(f"Predicted Rating: {pred:.2f} / 5.0")

# ==============================
# PAGE 4: RECOMMENDATIONS
# ==============================
elif page == "🎯 Get Recommendations":

    st.title("🎯 Personalized Recommendations")

    user_id = st.number_input("Enter User ID", min_value=1)

    if st.button("Get Recommendations"):

        if user_id in matrix.index:

            user_vector = matrix.loc[user_id].values.reshape(1, -1)
            similarity = matrix.dot(user_vector.T).sort_values(0, ascending=False)

            similar_users = similarity.index[1:6]

            recs = df[df['UserId'].isin(similar_users)] \
                .groupby(['Attraction'])['Rating'] \
                .mean().sort_values(ascending=False).head(5)

            st.dataframe(recs)

        else:
            st.warning("User not found in matrix.")