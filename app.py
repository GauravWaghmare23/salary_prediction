import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Salary Dashboard", layout="wide")

# -------------------------
# Load Model
# -------------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Custom CSS (Premium Look)
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding: 2rem;
}
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    color: white;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
}
.metric {
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar (Like Dashboard)
# -------------------------
st.sidebar.title("💼 Salary AI")
st.sidebar.markdown("### Navigation")

st.sidebar.button("Dashboard")
st.sidebar.button("Employees")
st.sidebar.button("Analytics")
st.sidebar.button("Settings")

st.sidebar.divider()

st.sidebar.markdown("### 👤 User")
st.sidebar.write("Gaurav Waghmare")

# -------------------------
# Top Header
# -------------------------
st.title("📊 Employee Salary Dashboard")

# -------------------------
# Layout
# -------------------------
col1, col2, col3 = st.columns([1,2,1])

# -------------------------
# Profile Card
# -------------------------
with col1:
    st.markdown("### 👤 Employee Profile")

    st.markdown("""
    <div class="card">
        <h3>Sample Employee</h3>
        <p>Role: Software Engineer</p>
        <p>Status: Active</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# KPI Cards
# -------------------------
with col2:
    st.markdown("### 📈 Performance Overview")

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric("Avg Salary", "₹8.5L", "+5%")

    with k2:
        st.metric("Experience", "5 Years", "+1")

    with k3:
        st.metric("Growth", "12%", "+2%")

# -------------------------
# Prediction Input
# -------------------------
st.divider()
st.markdown("## 🔍 Predict Salary")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    age = st.slider("Age", 18, 60, 25)

with c2:
    gender = st.selectbox("Gender", ["Male", "Female"])

with c3:
    education = st.selectbox("Education", ["Bachelor's", "Master's", "PhD"])

with c4:
    job = st.selectbox("Job", ["Software Engineer", "Data Scientist", "Manager"])

with c5:
    experience = st.slider("Experience", 0, 30, 1)

# -------------------------
# Prediction
# -------------------------
if st.button("🚀 Predict Salary"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education": [education],
        "Job": [job],
        "Experience": [experience]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Salary: ₹{prediction:,.0f}")

    # -------------------------
    # Chart Section
    # -------------------------
    st.markdown("### 📉 Salary Growth Trend")

    exp_range = np.arange(0, 30)

    chart_df = pd.DataFrame({
        "Experience": exp_range,
        "Salary": [
            model.predict(pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Education": [education],
                "Job": [job],
                "Experience": [e]
            }))[0]
            for e in exp_range
        ]
    })

    st.line_chart(chart_df.set_index("Experience"))

    # -------------------------
    # Bar Chart
    # -------------------------
    st.markdown("### 📊 Input Summary")

    feature_df = pd.DataFrame({
        "Feature": ["Age", "Experience"],
        "Value": [age, experience]
    })

    st.bar_chart(feature_df.set_index("Feature"))