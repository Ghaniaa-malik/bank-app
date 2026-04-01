import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Bank Intelligence", layout="wide")

st.title("💳 AI-Powered Bank Intelligence SaaS")

# =====================================================
# 1️⃣ UPLOAD DATA
# =====================================================
uploaded_file = st.file_uploader("📂 Upload Your CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Upload a dataset to generate AI insights")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset Loaded Successfully ✅")

# =====================================================
# BASIC CLEAN
# =====================================================
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

if "deposit" not in df.columns:
    st.error("❌ 'deposit' column missing in dataset")
    st.stop()

# =====================================================
# MODEL TRAINING
# =====================================================
X = df_encoded.drop("deposit", axis=1)
y = df_encoded["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# =====================================================
# AI INSIGHTS ENGINE 🤖
# =====================================================
st.header("🧠 AI Auto Insights Report")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("Avg Age", round(df["age"].mean(), 1) if "age" in df.columns else "N/A")
col3.metric("Avg Balance", round(df["balance"].mean(), 1) if "balance" in df.columns else "N/A")
col4.metric("Model Accuracy", f"{accuracy:.2f}")

st.markdown("### 🔍 Key Insights")

insights = []

if "age" in df.columns:
    if df["age"].mean() > 40:
        insights.append("Most customers are middle-aged → stable financial users")

if "balance" in df.columns:
    if df["balance"].mean() > 2000:
        insights.append("High average balance → strong deposit potential")

if "campaign" in df.columns:
    insights.append("Campaign activity impacts customer conversion")

if len(insights) == 0:
    insights.append("Dataset is balanced but needs deeper marketing analysis")

for i in insights:
    st.info("👉 " + i)

# =====================================================
# DASHBOARD
# =====================================================
st.header("📊 Smart Dashboard")

c1, c2 = st.columns(2)

with c1:
    if "age" in df.columns:
        st.plotly_chart(px.histogram(df, x="age", title="Age Distribution"), use_container_width=True)

with c2:
    if "balance" in df.columns:
        st.plotly_chart(px.histogram(df, x="balance", title="Balance Distribution"), use_container_width=True)

if "deposit" in df.columns:
    st.plotly_chart(px.pie(df, names="deposit", title="Deposit Ratio"), use_container_width=True)

# =====================================================
# RELATIONSHIP ANALYSIS
# =====================================================
st.header("📈 Relationship Analysis")

numeric_cols = df.select_dtypes(include=np.number).columns

if len(numeric_cols) >= 2:
    x_axis = st.selectbox("X-Axis", numeric_cols)
    y_axis = st.selectbox("Y-Axis", numeric_cols)

    st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis), use_container_width=True)

# =====================================================
# PREDICTION SYSTEM
# =====================================================
st.header("🧠 Customer Prediction System")

input_data = {}

for col in X.columns:

    if pd.api.types.is_numeric_dtype(df[col]):

        col_data = pd.to_numeric(df[col], errors='coerce')

        min_val = float(col_data.min())
        max_val = float(col_data.max())
        mean_val = float(col_data.mean())

        input_data[col] = st.number_input(
            col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

    else:
        input_data[col] = st.selectbox(col, df[col].dropna().unique())

if st.button("🚀 Predict"):

    input_df = pd.DataFrame([input_data])

    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = input_df[col].astype('category').cat.codes

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("✅ HIGH CHANCE: WILL SUBSCRIBE")
    else:
        st.error("❌ LOW CHANCE: WILL NOT SUBSCRIBE")

    st.progress(int(prob * 100))
    st.write(f"Confidence: {prob*100:.2f}%")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.header("🤖 AI Feature Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

st.plotly_chart(
    px.bar(importance_df.head(10),
           x="Importance",
           y="Feature",
           orientation="h",
           title="Top Driving Factors"),
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("🚀 AI SaaS Bank Intelligence System | Final Pro Version")
