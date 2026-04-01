# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Set Page Config
# ----------------------------
st.set_page_config(
    page_title="💳 AI-Powered Bank Marketing Intelligence",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# Title & Description
# ----------------------------
st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.markdown("""
Predict customer subscription using **Machine Learning**.  
Developed by **Ghania Iftikhar | Machine Learning IDS Project**.
""")

# ----------------------------
# Load Model Safely
# ----------------------------
FILE_DIR = Path(__file__).parent
model_path = FILE_DIR / "model.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ model.pkl not found! Make sure the file is in the project folder.")
    st.stop()

# ----------------------------
# Dataset Upload (Optional)
# ----------------------------
st.sidebar.header("Upload Dataset (Optional)")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
else:
    st.info("Upload a CSV to view data here.")

# ----------------------------
# Customer Input for Prediction
# ----------------------------
st.sidebar.header("Customer Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 95, 30)
    balance = st.sidebar.number_input("Account Balance", min_value=0, value=1000)
    duration = st.sidebar.number_input("Last Call Duration (seconds)", min_value=0, value=300)
    campaign = st.sidebar.number_input("Campaign Contacts", min_value=1, value=1)
    
    data = {
        "age": age,
        "balance": balance,
        "duration": duration,
        "campaign": campaign
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

st.write("### Customer Input Data")
st.dataframe(input_df)

# ----------------------------
# Prediction
# ----------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.write("### Prediction")
st.write("✅ Customer will subscribe" if prediction[0]==1 else "❌ Customer will NOT subscribe")

st.write("### Prediction Probability")
st.write(f"Subscribe: {prediction_proba[0][1]*100:.2f}% | Not Subscribe: {prediction_proba[0][0]*100:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Developed by **Ghania Iftikhar | Machine Learning IDS Project**")
