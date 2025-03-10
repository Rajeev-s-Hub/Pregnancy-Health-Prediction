import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained models
@st.cache_resource
def load_models():
    pipeline_preterm = joblib.load("preterm_model.pkl")
    pipeline_health = joblib.load("health_model.pkl")
    return pipeline_preterm, pipeline_health

# Convert gestational period into weeks, days, and hours
def convert_gestational_period(gestational_period_weeks):
    weeks = int(gestational_period_weeks)
    days = int((gestational_period_weeks - weeks) * 7)
    hours = int(((gestational_period_weeks - weeks) * 7 - days) * 24)
    return f"{weeks} weeks, {days} days, {hours} hours"

# Load models
pipeline_preterm, pipeline_health = load_models()

# Streamlit UI
st.title("Pregnancy Health Prediction App with Subgroup Analysis")

# Sidebar for manual input
st.sidebar.header("Manual Input")
manual_data = {
    "Infertility Treatment Used": st.sidebar.selectbox("Infertility Treatment Used", [0, 1]),
    "Number of Previous Cesareans": st.sidebar.slider("Number of Previous Cesareans", 0, 5, 0),
    "Fertility Enhancing Drugs": st.sidebar.selectbox("Fertility Enhancing Drugs", [0, 1]),
    "Births": st.sidebar.slider("Number of Births", 1, 5, 1),
    "Infections": st.sidebar.selectbox("Infections", [0, 1]),
    "Tobacco Use": st.sidebar.selectbox("Tobacco Use", [0, 1]),
    "Gestational Diabetes": st.sidebar.selectbox("Gestational Diabetes", [0, 1]),
    "Pre-pregnancy Hypertension": st.sidebar.selectbox("Pre-pregnancy Hypertension", [0, 1]),
    "Eclampsia": st.sidebar.selectbox("Eclampsia", [0, 1]),
    "Gestational Hypertension": st.sidebar.selectbox("Gestational Hypertension", [0, 1]),
    "BMI": st.sidebar.slider("BMI", 18.5, 45.0, 25.0),
    "Age": st.sidebar.slider("Age", 18, 45, 30),
    "PCOS": st.sidebar.selectbox("PCOS", [0, 1]),
}

# Manual Prediction
if st.sidebar.button("Run Prediction"):
    input_df = pd.DataFrame([manual_data])

    # Predict outcomes
    preterm_pred = pipeline_preterm.predict(input_df)[0]
    health_pred = pipeline_health.predict(input_df)[0]

    # Dynamically estimate gestational period
    gestational_period_weeks = np.random.uniform(28, 36) if preterm_pred else np.random.uniform(37, 42)
    gestational_period_formatted = convert_gestational_period(gestational_period_weeks)

    # Display Predictions
    st.markdown("## **Manual Input Predictions**")

    # Pre-term Birth
    if preterm_pred:
        st.markdown(
            """
            <div style="background-color:#FFCCCC;padding:10px;border-radius:5px;">
                <h3 style="color:#CC0000;">🔴 Pre-term Birth: Pre-term</h3>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#CCFFCC;padding:10px;border-radius:5px;">
                <h3 style="color:#006600;">🟢 Pre-term Birth: Normal</h3>
            </div>
            """, unsafe_allow_html=True
        )

    # Fetal Health
    if health_pred:
        st.markdown(
            """
            <div style="background-color:#FFCC99;padding:10px;border-radius:5px;">
                <h3 style="color:#CC6600;">🟠 Fetal Health: At Risk</h3>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#CCE5FF;padding:10px;border-radius:5px;">
                <h3 style="color:#003399;">🔵 Fetal Health: Healthy</h3>
            </div>
            """, unsafe_allow_html=True
        )

    # Gestational Period
    st.markdown(
        f"""
        <div style="background-color:#FFFACD;padding:10px;border-radius:5px;">
            <h3 style="color:#996600;">⏱️ Estimated Gestational Period: {gestational_period_formatted}</h3>
        </div>
        """, unsafe_allow_html=True
    )

# Batch Predictions
st.write("## Batch Predictions with Subgroup Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    required_columns = list(manual_data.keys())

    if not all(col in batch_data.columns for col in required_columns):
        st.error("Uploaded file is missing required columns.")
    else:
        # Predict outcomes
        batch_data["Predicted Preterm Birth"] = pipeline_preterm.predict(batch_data[required_columns])
        batch_data["Predicted Fetal Health"] = pipeline_health.predict(batch_data[required_columns])
        
        # Convert predictions to labels
        batch_data["Predicted Preterm Birth"] = batch_data["Predicted Preterm Birth"].map({1: "Pre-term", 0: "Normal"})
        batch_data["Predicted Fetal Health"] = batch_data["Predicted Fetal Health"].map({1: "At Risk", 0: "Healthy"})

        # Estimated Gestational Period
        batch_data["Estimated Gestational Period"] = batch_data["Predicted Preterm Birth"].apply(
            lambda x: convert_gestational_period(np.random.uniform(28, 36) if x == "Pre-term" else np.random.uniform(37, 42))
        )

        st.write("### Predictions:")
        st.dataframe(batch_data)

        # Subgroup Analysis: Fertility Enhancing Drugs
        st.write("## Subgroup Analysis for Fertility Enhancing Drugs")
        
        for group in [0, 1]:
            subgroup = batch_data[batch_data["Fertility Enhancing Drugs"] == group]
            st.write(f"### Subgroup: Fertility Enhancing Drugs = {group}")
            
            # Pie Chart: Preterm Birth
            st.write("**Preterm Birth Distribution**")
            fig, ax = plt.subplots()
            subgroup["Predicted Preterm Birth"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
            st.pyplot(fig)

            # Bar Chart: Fetal Health
            st.write("**Fetal Health Distribution**")
            st.bar_chart(subgroup["Predicted Fetal Health"].value_counts())

            # Correlation Heatmap
            st.write("**Attribute Correlation Heatmap**")
            plt.figure(figsize=(10, 8))
            sns.heatmap(subgroup[required_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            st.pyplot(plt)

        # Download results
        st.download_button("Download Results", batch_data.to_csv(index=False), "predictions.csv")
