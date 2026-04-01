# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="💳 AI Bank Marketing Intelligence", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.subheader("Predict Customer Subscription using Machine Learning")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv")

df = load_data()

# Show dataset info
st.markdown("### 📊 Dataset Overview")
st.write("Total Customers:", df.shape[0])
st.write("Total Features:", df.shape[1])

# Optional: Show visualizations
st.markdown("### 📈 Data Visualizations")
st.bar_chart(df['age'].value_counts().sort_index())

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Customer Prediction System
st.markdown("### 🧠 Customer Prediction System")
age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 30)
balance = st.number_input("Account Balance", float(df['balance'].min()), float(df['balance'].max()), 1000.0)
duration = st.number_input("Last Call Duration (seconds)", float(df['duration'].min()), float(df['duration'].max()), 300.0)
campaign = st.number_input("Number of Contacts in this Campaign", int(df['campaign'].min()), int(df['campaign'].max()), 1)

# Predict button
if st.button("Predict Subscription"):
    input_data = np.array([[age, balance, duration, campaign]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer is likely to subscribe! (Probability: {probability:.2f})")
    else:
        st.warning(f"❌ Customer is unlikely to subscribe. (Probability: {probability:.2f})")

st.markdown("---")
st.markdown("Developed by **Ghania Iftikhar** | Machine Learning IDS Project")
