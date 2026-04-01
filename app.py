# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Bank Marketing AI", layout="wide")
st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.write("Predict Customer Subscription using Machine Learning")

# ---------------- Dataset Load ----------------
st.header("📊 Dataset Overview")
try:
    df = pd.read_csv("bank.csv")  # make sure bank.csv is in the same folder
    st.success("Dataset Loaded Successfully!")
    st.write(f"**Total Customers:** {df.shape[0]}")
    st.write(f"**Total Features:** {df.shape[1]}")
    st.write("Columns in Dataset:")
    st.write(list(df.columns))
except FileNotFoundError:
    st.error("bank.csv not found! Upload your dataset in the same folder as app.py")
    st.stop()

# ---------------- Select Target ----------------
target_column = st.selectbox("Select the target column", df.columns)
st.write(f"Target column selected: **{target_column}**")

# ---------------- Select Features ----------------
feature_columns = st.multiselect(
    "Select feature columns", [col for col in df.columns if col != target_column]
)

# ---------------- Train Model ----------------
st.header("⚡ Train Machine Learning Model")
if st.button("Train Model"):
    if not feature_columns:
        st.warning("Please select at least one feature column!")
    else:
        X = df[feature_columns]
        y = df[target_column]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success("Model trained successfully!")

        # Save model
        joblib.dump(model, "model.pkl")
        st.info("Model saved as model.pkl")

        # Show accuracy
        acc = model.score(X_test, y_test)
        st.write(f"Model Accuracy: **{acc*100:.2f}%**")

# ---------------- Prediction ----------------
st.header("🔹 Customer Prediction System")
try:
    model = joblib.load("model.pkl")
    st.success("Model loaded successfully for prediction!")

    # Dynamically generate inputs based on features
    input_data = {}
    for feature in feature_columns:
        if df[feature].dtype in ['int64', 'float64']:
            input_data[feature] = st.number_input(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].median()))
        else:
            input_data[feature] = st.selectbox(feature, df[feature].unique())

    if st.button("Predict"):
        # Prepare input for prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f"Predicted Subscription: **{prediction[0]}**")

except FileNotFoundError:
    st.warning("model.pkl not found! Train the model first.")
