import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Bank AI Dashboard", layout="wide")

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv")

df = load_data()

# -----------------------
# ENCODE DATA
# -----------------------
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# -----------------------
# FEATURES & TARGET
# -----------------------
X = df_encoded.drop("deposit", axis=1)
y = df_encoded["deposit"]

# -----------------------
# TRAIN MODEL
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Data Analysis", "Prediction", "Model Insights", "Bulk Prediction"]
)

# -----------------------
# HEADER
# -----------------------
st.title("💳 AI-Powered Bank Marketing Intelligence System")

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":

    st.header("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Avg Age", int(df["age"].mean()))
    col3.metric("Model Accuracy", f"{accuracy:.2f}")

    st.subheader("📈 Insights")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x="age", title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x="balance", title="Balance Distribution")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# DATA ANALYSIS
# =====================================================
elif page == "Data Analysis":

    st.header("📈 Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Correlation Heatmap")
    corr = df_encoded.corr()

    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Deposit Ratio")
    fig = px.pie(df, names="deposit", title="Subscription Ratio")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age vs Balance")
    fig = px.scatter(df, x="age", y="balance")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================
elif page == "Prediction":

    st.header("🧠 Customer Prediction")

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

            st.subheader("Confidence Level")
            st.progress(int(probability * 100))
            st.write(f"Probability: {probability*100:.2f}%")

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
        orientation="h",
        title="Top Important Features"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# BULK PREDICTION
# =====================================================
elif page == "Bulk Prediction":

    st.header("📂 Bulk Customer Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        new_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(new_data.head())

        required_cols = X.columns.tolist()

        missing_cols = [col for col in required_cols if col not in new_data.columns]

        if len(missing_cols) > 0:
            st.error(f"Missing columns: {missing_cols}")
        else:
            new_data_encoded = new_data.copy()

            for col in new_data_encoded.select_dtypes(include='object').columns:
                new_data_encoded[col] = new_data_encoded[col].astype('category').cat.codes

            new_data_encoded = new_data_encoded[required_cols]

            predictions = model.predict(new_data_encoded)
            probs = model.predict_proba(new_data_encoded)[:,1]

            new_data["Prediction"] = np.where(predictions==1, "Yes", "No")
            new_data["Confidence"] = probs

            st.subheader("Results")
            st.dataframe(new_data)

            csv = new_data.to_csv(index=False).encode('utf-8')

            st.download_button(
                "⬇ Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Developed by Ghania Iftikhar | IDS Project")
