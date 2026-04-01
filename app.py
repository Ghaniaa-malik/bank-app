import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Bank Marketing App", layout="centered")

st.title("📊 Bank Term Deposit Prediction App")

st.write("This app predicts whether a customer will subscribe to a term deposit.")

# Load data
df = pd.read_csv("bank.csv")

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

# Features and target
X = df.drop('deposit', axis=1)
y = df['deposit']

# Normalize
X = (X - X.min()) / (X.max() - X.min())

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 95, 30)
balance = st.number_input("Balance", -2000, 100000, 1000)
duration = st.number_input("Duration", 0, 5000, 300)
campaign = st.number_input("Campaign", 1, 50, 1)

input_data = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "duration": [duration],
    "campaign": [campaign]
})

for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Customer WILL Subscribe")
    else:
        st.error("❌ Customer will NOT Subscribe")
