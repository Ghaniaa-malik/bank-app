import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Bank Dashboard", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")

# ---------------------------
# STEP 1: FILE UPLOAD FIRST
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload Your Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue")
    st.stop()

# ---------------------------
# LOAD USER DATA
# ---------------------------
df = pd.read_csv(uploaded_file)

st.success("✅ File Uploaded Successfully")

st.subheader("📊 Your Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# CHECK REQUIRED COLUMN
# ---------------------------
if "deposit" not in df.columns:
    st.error("❌ Your dataset must contain 'deposit' column")
    st.stop()

# ---------------------------
# ENCODE DATA
# ---------------------------
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# ---------------------------
# FEATURES & TARGET
# ---------------------------
X = df_encoded.drop("deposit", axis=1)
y = df_encoded["deposit"]

# ---------------------------
# TRAIN MODEL
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Data Analysis", "Prediction", "Model Insights"]
)

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":

    st.header("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Model Accuracy", f"{accuracy:.2f}")

    st.subheader("📈 Age Distribution")

    if "age" in df.columns:
        fig = px.histogram(df, x="age")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💰 Balance Distribution")

    if "balance" in df.columns:
        fig = px.histogram(df, x="balance")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# DATA ANALYSIS
# =====================================================
elif page == "Data Analysis":

    st.header("📈 Data Analysis")

    st.subheader("Correlation Heatmap")
    fig = px.imshow(df_encoded.corr(), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    if "deposit" in df.columns:
        st.subheader("Deposit Ratio")
        fig = px.pie(df, names="deposit")
        st.plotly_chart(fig, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) >= 2:
        st.subheader("Scatter Plot")
        x_axis = st.selectbox("X-axis", numeric_cols)
        y_axis = st.selectbox("Y-axis", numeric_cols)

        fig = px.scatter(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================
elif page == "Prediction":

    st.header("🧠 Prediction System")

    input_data = {}

    for col in X.columns:
        if df[col].dtype == 'object':
            input_data[col] = st.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

    if st.button("🚀 Predict"):

        input_df = pd.DataFrame([input_data])

        for col in input_df.select_dtypes(include='object').columns:
            input_df[col] = input_df[col].astype('category').cat.codes

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("✅ WILL Subscribe")
        else:
            st.error("❌ Will NOT Subscribe")

        st.progress(int(prob * 100))
        st.write(f"Confidence: {prob*100:.2f}%")

# =====================================================
# MODEL INSIGHTS
# =====================================================
elif page == "Model Insights":

    st.header("🤖 Model Insights")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Developed by Ghania Iftikhar | IDS Project")
