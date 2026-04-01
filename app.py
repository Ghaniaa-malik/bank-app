# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

st.set_page_config(
    page_title="AI-Powered Bank Marketing Intelligence System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.subheader("Predict Customer Subscription using Machine Learning")

# ---------------- Dataset Upload ----------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully!")
    st.write("### Dataset Overview")
    st.write(f"Total Customers: {df.shape[0]}")
    st.write(f"Total Features: {df.shape[1]}")
    st.dataframe(df.head())

    st.write("### Columns in Dataset")
    st.write(df.columns)

    # Check if target column exists
    target_column = st.selectbox("Select the target column", df.columns, index=len(df.columns)-1)
    if target_column:
        st.success(f"Target column selected: {target_column}")

        # ---------------- Train Model ----------------
        st.write("### Train Machine Learning Model")
        features = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])
        if len(features) > 0:
            if st.button("Train Model"):
                X = df[features]
                y = df[target_column].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                st.success("Model Trained Successfully!")

                # Save the model
                joblib.dump(model, "model.pkl")
                st.info("Model saved as model.pkl")

        # ---------------- Prediction ----------------
        import os

if not os.path.exists("model.pkl"):
    st.warning("Model file not found! Please train the model first.")
else:
    model = joblib.load("model.pkl")
        st.write("### Customer Prediction System")
        if os.path.exists("model.pkl"):
            model = joblib.load("model.pkl")
            st.success("Model Loaded!")

            st.write("Enter Customer Details for Prediction")
            input_data = {}
            for feature in features:
                val = st.number_input(f"{feature}", value=float(df[feature].median()))
                input_data[feature] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                st.write(f"**Prediction:** {'Subscribed' if prediction==1 else 'Not Subscribed'}")
                st.write(f"**Probability of Subscription:** {probability:.2f}")
else:
    st.info("Please upload a CSV dataset to continue.")
