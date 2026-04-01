import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Bank Dashboard", layout="wide")

st.title("💳 AI-Powered Bank Marketing Intelligence System")

# =====================================================
# 1️⃣ UPLOAD FILE FIRST
# =====================================================
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("File uploaded successfully ✅")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =====================================================
# CHECK TARGET COLUMN
# =====================================================
if "deposit" not in df.columns:
    st.error("❌ Dataset must contain 'deposit' column")
    st.stop()

# =====================================================
# CLEAN + ENCODE
# =====================================================
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

X = df_encoded.drop("deposit", axis=1)
y = df_encoded["deposit"]

# =====================================================
# TRAIN MODEL
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Data Analysis", "Prediction", "Model Insights"]
)

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Accuracy", f"{accuracy:.2f}")

    if "age" in df.columns:
        st.subheader("Age Distribution")
        st.plotly_chart(px.histogram(df, x="age"), use_container_width=True)

    if "balance" in df.columns:
        st.subheader("Balance Distribution")
        st.plotly_chart(px.histogram(df, x="balance"), use_container_width=True)

# =====================================================
# DATA ANALYSIS
# =====================================================
elif page == "Data Analysis":

    st.header("📈 Data Analysis")

    st.subheader("Correlation Heatmap")
    st.plotly_chart(px.imshow(df_encoded.corr(), text_auto=True), use_container_width=True)

    if "deposit" in df.columns:
        st.subheader("Deposit Ratio")
        st.plotly_chart(px.pie(df, names="deposit"), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) >= 2:
        st.subheader("Scatter Plot")
        x_axis = st.selectbox("X-axis", numeric_cols)
        y_axis = st.selectbox("Y-axis", numeric_cols)

        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis), use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================
elif page == "Prediction":

    st.header("🧠 Prediction System")

    input_data = {}

    for col in X.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            col_data = pd.to_numeric(df[col], errors='coerce')

            min_val = float(col_data.min())
            max_val = float(col_data.max())
            mean_val = float(col_data.mean())

            if np.isnan(mean_val):
                mean_val = 0.0

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
            st.success("✅ WILL SUBSCRIBE")
        else:
            st.error("❌ WILL NOT SUBSCRIBE")

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

    st.plotly_chart(
        px.bar(importance_df.head(10),
               x="Importance",
               y="Feature",
               orientation="h"),
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("AI Bank Dashboard | Final Stable Version")
