import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Bank AI System", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.markdown("### Predict Customer Subscription using Machine Learning")

# Load dataset
df = pd.read_csv("bank.csv")

st.success("📊 Dataset Loaded Successfully!")
st.write(f"Total Customers: {df.shape[0]}")
st.write(f"Total Features: {df.shape[1]}")

# -------------------------------
# FIXED TARGET
# -------------------------------
target = "deposit"

# Encode categorical columns
label_encoders = {}

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & target
X = df.drop(target, axis=1)
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (NO model.pkl needed 🔥)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
st.write(f"### ✅ Model Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# DASHBOARD
# -------------------------------
st.subheader("📊 Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("Avg Age", int(df["age"].mean()))
col3.metric("Avg Balance", int(df["balance"].mean()))

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.subheader("🧠 Customer Prediction System")

user_input = {}

for col in X.columns:
    if col in label_encoders:
        user_input[col] = st.selectbox(col, label_encoders[col].classes_)
    else:
        user_input[col] = st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

# Prediction
if st.button("🚀 Predict"):

    input_df = pd.DataFrame([user_input])

    # Encode user input
    for col in label_encoders:
        if col in input_df:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("✅ HIGH CHANCE: Customer WILL Subscribe")
    else:
        st.error("❌ LOW CHANCE: Customer will NOT Subscribe")

    st.subheader("📊 Prediction Confidence")
    st.progress(float(prob))
    st.write(f"Probability: {prob*100:.2f}%")

# Footer
st.markdown("---")
st.caption("🚀 Developed by Ghania Iftikhar | IDS Project")
