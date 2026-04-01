import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="AI Bank Marketing Dashboard", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.markdown("Predict customer subscription using Machine Learning")

# Load data
df = pd.read_csv("bank.csv")

# Encode categorical columns
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

# -------------------
# Dashboard section
# -------------------

st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Total Features", len(df.columns))

st.dataframe(df.head())

# -------------------
# Charts
# -------------------

st.subheader("📈 Data Visualizations")

chart1, chart2 = st.columns(2)

with chart1:
    fig = plt.figure()
    plt.hist(df['age'], bins=20)
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    st.pyplot(fig)

with chart2:
    fig = plt.figure()
    df['deposit'].value_counts().plot(kind='bar')
    plt.title("Deposit Subscription Count")
    st.pyplot(fig)

# -------------------
# Prediction section
# -------------------

st.subheader("🧠 Customer Prediction System")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 95, 30)
    balance = st.number_input("Account Balance", -2000, 100000, 1000)
    duration = st.number_input("Call Duration", 0, 5000, 300)
    campaign = st.number_input("Campaign Contacts", 1, 50, 1)

with col2:

    if st.button("🚀 Predict Customer Behavior"):

        input_data = pd.DataFrame({
            "age":[age],
            "balance":[balance],
            "duration":[duration],
            "campaign":[campaign]
        })

        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[X.columns]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success("✅ HIGH CHANCE: Customer WILL Subscribe")
        else:
            st.error("❌ LOW CHANCE: Customer will NOT Subscribe")

        st.write("Prediction Confidence:")
        st.progress(float(probability))

st.markdown("---")
st.caption("Developed by Ghania Iftikhar | Machine Learning IDS Project")
