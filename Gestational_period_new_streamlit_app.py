import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

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

# Helper: Flatten a nested dictionary (keys joined by an underscore)
def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# Load models
pipeline_preterm, pipeline_health = load_models()

# Streamlit UI
st.title("Pregnancy Health Prediction App with Subgroup Analysis")

# Sidebar for manual input
st.sidebar.header("Manual Input")

# Add start date input for pregnancy
start_date = st.sidebar.date_input("Start Date of Pregnancy")

# Define manual input dictionary in the specified order
manual_data = {
    "Age": st.sidebar.slider("Age", 18, 45, 30),
    "BMI": st.sidebar.slider("BMI", 18.5, 45.0, 25.0),
    "PCOS": st.sidebar.radio("PCOS", ["No", "Yes"]) == "Yes",
    "Number of Previous Cesareans": st.sidebar.slider("Number of Previous Cesareans", 0, 5, 0),
    "Births": st.sidebar.slider("Number of Births", 1, 5, 1),
    "Infections": st.sidebar.radio("Infections (Hepatitis, COVID19, etc.)", ["No", "Yes"]) == "Yes",
    "Tobacco Use": st.sidebar.radio("Tobacco Use", ["No", "Yes"]) == "Yes",
    "Gestational Diabetes": st.sidebar.radio("Gestational Diabetes", ["No", "Yes"]) == "Yes",
    "Gestational Hypertension": st.sidebar.radio("Gestational Hypertension", ["No", "Yes"]) == "Yes",
    "Pre-pregnancy Hypertension": st.sidebar.radio("Pre-pregnancy Hypertension", ["No", "Yes"]) == "Yes",
    "Eclampsia": st.sidebar.radio("Eclampsia", ["No", "Yes"]) == "Yes",
    # Nested dictionary for Infertility Treatment
    "Infertility Treatment": {
         "IVF": st.sidebar.radio("IVF Treatment Used?", ["No", "Yes"]) == "Yes",
         "Clomifene citrate": st.sidebar.radio("Clomifene Citrate Used?", ["No", "Yes"]) == "Yes",
         "Others": st.sidebar.radio("Other Treatments Used?", ["No", "Yes"]) == "Yes",
    },
    # Nested dictionary for Fertility Enhancing Drugs
    "Fertility Enhancing Drugs": {
         "Letrozole": st.sidebar.radio("Letrozole Used?", ["No", "Yes"]) == "Yes",
         "Gonadotropins": st.sidebar.radio("Gonadotropins Used?", ["No", "Yes"]) == "Yes",
         "GnRH Analogues": st.sidebar.radio("GnRH Analogues Used?", ["No", "Yes"]) == "Yes",
         "Bromocriptine": st.sidebar.radio("Bromocriptine Used?", ["No", "Yes"]) == "Yes",
         "Cabergoline": st.sidebar.radio("Cabergoline Used?", ["No", "Yes"]) == "Yes",
         "Progesterone Support": st.sidebar.radio("Progesterone Support Used?", ["No", "Yes"]) == "Yes",
         "ART Adjuncts": st.sidebar.radio("ART Adjuncts Used?", ["No", "Yes"]) == "Yes",
    }
}

# Flatten the nested dictionary so that the model receives a flat feature set.
flat_manual_data = flatten_dict(manual_data)
required_columns = list(flat_manual_data.keys())

# Manual Prediction
if st.sidebar.button("Run Prediction"):
    input_df = pd.DataFrame([flat_manual_data]).astype(int)

    # Predict outcomes using the models
    preterm_pred = pipeline_preterm.predict(input_df)[0]
    health_pred = pipeline_health.predict(input_df)[0]

    # Dynamically estimate gestational period (in weeks)
    gestational_period_weeks = np.random.uniform(28, 36) if preterm_pred else np.random.uniform(37, 42)
    gestational_period_formatted = convert_gestational_period(gestational_period_weeks)

    # Calculate predicted due date based on start date and gestational period
    predicted_due_date = start_date + timedelta(days=int(gestational_period_weeks * 7))
    
    # Also calculate an approximate duration in months, weeks, days from the start date
    total_days = int(gestational_period_weeks * 7)
    months = total_days // 30  # approximate months
    remaining_days = total_days % 30
    weeks = remaining_days // 7
    days = remaining_days % 7
    duration_str = f"{months} months, {weeks} weeks, {days} days"

    # Display Predictions
    st.markdown("## **Manual Input Predictions**")
    
    # Birth Term Prediction (Pre-term vs. Full-term)
    if preterm_pred:
        st.markdown(
            """
            <div style="background-color:#FFCCCC;padding:10px;border-radius:5px;">
                <h3 style="color:#CC0000;">🔴 Birth Term: Pre-term</h3>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#CCFFCC;padding:10px;border-radius:5px;">
                <h3 style="color:#006600;">🟢 Birth Term: Full-term</h3>
            </div>
            """, unsafe_allow_html=True
        )
    
    # Fetal Health Prediction
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
    
    # Gestational Period Display
    st.markdown(
        f"""
        <div style="background-color:#FFFACD;padding:10px;border-radius:5px;">
            <h3 style="color:#996600;">⏱️ Estimated Gestational Period: {gestational_period_formatted}</h3>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Display Predicted Due Date and Duration from Start Date
    st.markdown(
        f"""
        <div style="background-color:#E0F7FA;padding:10px;border-radius:5px;">
            <h3 style="color:#00796B;">📅 Predicted Due Date: {predicted_due_date.strftime('%B %d, %Y')}</h3>
            <h4 style="color:#004D40;">(Approximately {duration_str} from start)</h4>
        </div>
        """, unsafe_allow_html=True
    )

# Batch Predictions Section
st.write("## Batch Predictions with Subgroup Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    
    # Ensure that the uploaded CSV has the required flattened keys
    if not all(col in batch_data.columns for col in required_columns):
        st.error("Uploaded file is missing required columns.")
    else:
        # Predict outcomes on the batch data
        batch_data["Predicted Birth Term"] = pipeline_preterm.predict(batch_data[required_columns])
        batch_data["Predicted Fetal Health"] = pipeline_health.predict(batch_data[required_columns])
        
        # Map predictions to labels
        batch_data["Predicted Birth Term"] = batch_data["Predicted Birth Term"].map({1: "Pre-term", 0: "Full-term"})
        batch_data["Predicted Fetal Health"] = batch_data["Predicted Fetal Health"].map({1: "At Risk", 0: "Healthy"})
        
        # Estimated Gestational Period for each prediction
        batch_data["Estimated Gestational Period"] = batch_data["Predicted Birth Term"].apply(
            lambda x: convert_gestational_period(np.random.uniform(28, 36) if x == "Pre-term" else np.random.uniform(37, 42))
        )
        
        st.write("### Predictions:")
        st.dataframe(batch_data)
        
        # Subgroup Analysis for Fertility Enhancing Drugs (example: Letrozole subgroup)
        st.write("## Subgroup Analysis for Fertility Enhancing Drugs")
        for group in [0, 1]:
            subgroup = batch_data[batch_data["Fertility Enhancing Drugs_Letrozole"] == group]
            if not subgroup.empty:
                st.write(f"### Subgroup: Letrozole Used = {group}")
                # Pie Chart: Birth Term Distribution
                st.write("**Birth Term Distribution**")
                fig, ax = plt.subplots()
                subgroup["Predicted Birth Term"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
                st.pyplot(fig)
                # Bar Chart: Fetal Health Distribution
                st.write("**Fetal Health Distribution**")
                st.bar_chart(subgroup["Predicted Fetal Health"].value_counts())
                # Correlation Heatmap for the subgroup
                st.write("**Attribute Correlation Heatmap**")
                plt.figure(figsize=(10, 8))
                sns.heatmap(subgroup[required_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
                st.pyplot(plt)
        
        # Download batch prediction results
        st.download_button("Download Results", batch_data.to_csv(index=False), "predictions.csv")
