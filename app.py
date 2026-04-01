import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="💳 AI Bank Marketing Dashboard", layout="wide")
st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.subheader("Predict Customer Subscription using Machine Learning")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank.csv")  # Make sure bank.csv is uploaded
    return df

df = load_data()

# Dataset Overview
st.markdown("### 📊 Dataset Overview")
st.write(f"Total Customers: {df.shape[0]}")
st.write(f"Total Features: {df.shape[1]}")

# Safe Data Visualization
st.markdown("### 📈 Age Distribution")
if 'age' in df.columns:
    st.bar_chart(df['age'].clip(0).value_counts().sort_index())  # clip negative ages if any

st.markdown("### 🔹 Customer Prediction System")

# Sidebar Inputs
age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 30)
balance = st.number_input("Account Balance", float(df['balance'].min()), float(df['balance'].max()), 1000.0)
duration = st.number_input("Last Call Duration (seconds)", float(df['duration'].min()), float(df['duration'].max()), 300.0)
campaign = st.number_input("Number of Contacts in this Campaign", int(df['campaign'].min()), int(df['campaign'].max()), 1)

# Train simple model inside app
@st.cache_resource
def train_model():
    X = df[['age','balance','duration','campaign']]
    y = df['y']  # make sure target column exists
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Prediction
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
