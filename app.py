# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

st.set_page_config(page_title="AI-Powered Bank Marketing Intelligence", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.subheader("Predict Customer Subscription using Machine Learning")

# ---- Dataset Loading ----
DATA_PATH = "bank.csv"
MODEL_PATH = "model.pkl"

@st.cache_data
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at path: {path}")
        return None

df = load_dataset(DATA_PATH)

if df is not None:
    st.success("📊 Dataset Loaded Successfully!")
    st.write(f"**Total Customers:** {df.shape[0]}")
    st.write(f"**Total Features:** {df.shape[1]}")
    st.write("**Columns in Dataset:**")
    st.write(list(df.columns))
    
    # ---- Target Column ----
    target_col = "deposit"
    st.write(f"**Target column selected:** {target_col}")
    
    # ---- Feature Selection ----
    feature_cols = [col for col in df.columns if col != target_col]
    st.write("**Feature columns:**", feature_cols)

    # ---- Encode Categorical Columns ----
    @st.cache_data
    def encode_data(df):
        df_encoded = df.copy()
        label_encoders = {}
        for col in df_encoded.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
        return df_encoded, label_encoders

    df_encoded, label_encoders = encode_data(df)

    # ---- Split Data ----
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Train or Load Model ----
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        st.info("✅ Loaded existing trained model.")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        st.success("✅ Model trained and saved successfully.")

    # ---- Model Accuracy ----
    accuracy = model.score(X_test, y_test)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    # ---- Customer Prediction ----
    st.write("### 🔹 Customer Prediction System")
    user_input = {}
    for col in feature_cols:
        if df[col].dtype == 'object':
            user_input[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    
    if st.button("Predict Subscription"):
        # Encode input
        input_df = pd.DataFrame([user_input])
        for col in input_df.select_dtypes(include=['object']).columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])
        
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0][1]
        
        st.write(f"### Prediction: **{'Subscribed' if prediction==1 else 'Not Subscribed'}**")
        st.write(f"### Probability of Subscription: {prediction_prob*100:.2f}%")
else:
    st.error("Cannot proceed without dataset.")
