import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Bank Marketing Intelligence System",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("bank.csv")

# Load trained ML model
model = joblib.load("model.pkl")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Data Analysis",
        "Customer Prediction",
        "Model Insights",
        "Bulk Prediction"
    ]
)

# -------------------------------
# Header
# -------------------------------
st.title("💳 AI-Powered Bank Marketing Intelligence System")

st.markdown("""
Predict customer subscription to bank term deposits using machine learning.
This system analyzes client attributes such as age, balance, campaign contacts,
and call duration to determine subscription likelihood.
""")

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Average Age", int(df["age"].mean()))
    col3.metric("Average Balance", int(df["balance"].mean()))

    st.subheader("Age Distribution")

    fig = px.histogram(df, x="age")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Balance Distribution")

    fig = px.histogram(df, x="balance")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# DATA ANALYSIS
# =====================================================

elif page == "Data Analysis":

    st.header("📈 Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age vs Balance")

    fig = px.scatter(df, x="age", y="balance")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# CUSTOMER PREDICTION
# =====================================================

elif page == "Customer Prediction":

    st.header("🧠 Customer Prediction System")

    age = st.slider("Age", 18, 95, 30)
    balance = st.number_input("Account Balance", value=1000)
    duration = st.number_input("Call Duration", value=300)
    campaign = st.number_input("Campaign Contacts", value=1)

    if st.button("Predict Subscription"):

        input_data = np.array([[age, balance, duration, campaign]])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success("Customer likely to SUBSCRIBE")
        else:
            st.error("Customer unlikely to subscribe")

        st.subheader("Subscription Probability")

        st.progress(int(probability * 100))

        st.write(f"Probability: {probability*100:.2f}%")

# =====================================================
# MODEL INSIGHTS
# =====================================================

elif page == "Model Insights":

    st.header("🤖 Model Insights")

    st.subheader("Feature Importance")

    importance = model.feature_importances_

    features = ["age", "balance", "duration", "campaign"]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
Model Used: Random Forest Classifier

This model analyzes customer attributes and predicts whether the client will subscribe to a bank deposit campaign.
""")

# =====================================================
# BULK CSV PREDICTION
# =====================================================

elif page == "Bulk Prediction":

    st.header("📂 Bulk Customer Prediction")

    uploaded_file = st.file_uploader("Upload CSV File")

    if uploaded_file:

        new_data = pd.read_csv(uploaded_file)

        predictions = model.predict(new_data)

        new_data["Prediction"] = predictions

        st.subheader("Prediction Results")

        st.dataframe(new_data)

        csv = new_data.to_csv(index=False)

        st.download_button(
            "Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.caption("Developed by Ghania Iftikhar | Machine Learning IDS Project")
