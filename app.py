import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Bank AI Dashboard", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.markdown("### Predict Customer Subscription using Machine Learning")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv")

df = load_data()

st.success("✅ Dataset Loaded Successfully")

# -------------------------
# Encode Data (IMPORTANT FIX)
# -------------------------
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# -------------------------
# Features & Target
# -------------------------
X = df_encoded.drop("deposit", axis=1)
y = df_encoded["deposit"]

# -------------------------
# Train Model
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -------------------------
# DASHBOARD
# -------------------------
st.subheader("📊 Dashboard Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Average Age", int(df["age"].mean()))
col3.metric("Model Accuracy", f"{accuracy:.2f}")

# -------------------------
# Charts
# -------------------------
st.subheader("📈 Data Visualizations")

c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(df, x="age", title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.histogram(df, x="balance", title="Balance Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Deposit Count
fig = px.pie(df, names="deposit", title="Deposit Subscription Ratio")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Prediction Section
# -------------------------
st.subheader("🧠 Customer Prediction System")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 95, 30)
    balance = st.number_input("Balance", -2000, 100000, 1000)
    duration = st.number_input("Call Duration", 0, 5000, 300)
    campaign = st.number_input("Campaign Contacts", 1, 50, 1)

with col2:
    if st.button("🚀 Predict"):

        input_dict = {col: 0 for col in X.columns}

        input_dict["age"] = age
        input_dict["balance"] = balance
        input_dict["duration"] = duration
        input_dict["campaign"] = campaign

        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("✅ HIGH CHANCE: Customer WILL Subscribe")
        else:
            st.error("❌ LOW CHANCE: Customer will NOT Subscribe")

        st.subheader("📊 Confidence Level")
        st.progress(int(probability * 100))
        st.write(f"Probability: {probability*100:.2f}%")

# -------------------------
# Feature Importance
# -------------------------
st.subheader("🤖 Model Insights")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

fig = px.bar(
    importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Important Features"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Developed by Ghania Iftikhar | Machine Learning IDS Project")
