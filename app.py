import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Bank AI System", layout="wide")


st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3, h4 {
    color: #00FFAA;
}
.stButton>button {
    background-color: #00FFAA;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #00cc88;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.markdown("### Predict Customer Subscription with Machine Learning")
st.markdown("---")

# Load data
df = pd.read_csv("bank.csv")

# Encode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

X = df.drop('deposit', axis=1)
y = df['deposit']

# Normalize
X = (X - X.min()) / (X.max() - X.min())

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Information")

    age = st.slider("Age", 18, 95, 30)
    balance = st.number_input("Account Balance", -2000, 100000, 1000)
    duration = st.number_input("Contact Duration", 0, 5000, 300)
    campaign = st.number_input("Campaign Contacts", 1, 50, 1)

with col2:
    st.subheader("🧠 Prediction Engine")

    st.write("Click below to predict:")

    if st.button("🚀 Predict Customer Behavior"):

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

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ HIGH CHANCE: Customer WILL Subscribe")
        else:
            st.error("❌ LOW CHANCE: Customer will NOT Subscribe")

# Footer
st.markdown("---")
st.caption("🚀 Developed by Ghania Iftikhar | Machine Learning Project")
