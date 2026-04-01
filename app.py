import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Bank AI Dashboard", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")
st.markdown("### Advanced Customer Subscription Prediction Dashboard")

# Load data
df = pd.read_csv("bank.csv")

# Encode categorical
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features
X = df.drop("deposit", axis=1)
y = df["deposit"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)

# =========================
# DASHBOARD
# =========================
st.subheader("📊 Dashboard Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(df))
c2.metric("Avg Age", int(df["age"].mean()))
c3.metric("Model Accuracy", f"{acc*100:.2f}%")

# =========================
# CHARTS
# =========================
st.subheader("📈 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="age", title="Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x="balance", title="Balance Distribution")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = px.box(df, x="deposit", y="balance", title="Balance vs Deposit")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.scatter(df, x="age", y="balance", color="deposit",
                      title="Age vs Balance")
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# PREDICTION
# =========================
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

if st.button("🚀 Predict Customer Behavior"):
    input_df = pd.DataFrame([user_input])

    for col in label_encoders:
        if col in input_df:
            input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("✅ HIGH CHANCE: Customer WILL Subscribe")
    else:
        st.error("❌ LOW CHANCE: Customer will NOT Subscribe")

    st.subheader("📊 Prediction Confidence")
    st.progress(float(prob))
    st.write(f"Probability: {prob*100:.2f}%")

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("🤖 Model Insights")

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance")

fig5 = px.bar(imp_df, x="Importance", y="Feature",
              orientation="h", title="Feature Importance")

st.plotly_chart(fig5, use_container_width=True)

# Footer
st.markdown("---")
st.caption("🚀 Developed by Ghania Iftikhar | AI ML Dashboard")
