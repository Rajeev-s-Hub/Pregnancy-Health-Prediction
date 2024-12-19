import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load pre-trained models
@st.cache_resource
def load_models():
    pipeline_preterm = joblib.load("preterm_model.pkl")
    pipeline_health = joblib.load("health_model.pkl")
    return pipeline_preterm, pipeline_health

# Connect to MySQL
def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Rajdna0510@#$%",
            database="PregnancyData"
        )
        return connection
    except Error as err:
        st.error(f"Database connection error: {err}")
        return None

# Insert data into MySQL
def insert_prediction(cursor, user_input, preterm_prediction, health_prediction, gestational_period):
    try:
        query = """
        INSERT INTO Predictions (user_input, predicted_preterm, predicted_health, gestational_period)
        VALUES (%s, %s, %s, %s);
        """
        cursor.execute(query, (user_input, preterm_prediction, health_prediction, gestational_period))
    except Error as err:
        st.error(f"Failed to insert prediction: {err}")

# Convert gestational period into weeks, days, and hours
def convert_gestational_period(gestational_period_weeks):
    weeks = int(gestational_period_weeks)
    days = int((gestational_period_weeks - weeks) * 7)
    hours = int(((gestational_period_weeks - weeks) * 7 - days) * 24)
    return f"{weeks} weeks, {days} days, {hours} hours"

# Load models
pipeline_preterm, pipeline_health = load_models()

# Streamlit UI
st.title("Pregnancy Health Prediction App with Enhanced Display")

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

# Establish a persistent database connection
connection = create_db_connection()

if not connection:
    st.error("Unable to connect to the database. Please check the connection details.")
else:
    cursor = connection.cursor()

# Manual Prediction
if st.sidebar.button("Run Prediction"):
    input_df = pd.DataFrame([manual_data])

    # Predict outcomes
    preterm_pred = pipeline_preterm.predict(input_df)[0]
    health_pred = pipeline_health.predict(input_df)[0]

    # Dynamically estimate gestational period
    gestational_period_weeks = np.random.uniform(28, 36) if preterm_pred else np.random.uniform(37, 42)
    gestational_period_formatted = convert_gestational_period(gestational_period_weeks)

    # Enhanced Display with Styled Markdown
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

    # Save to MySQL
    if connection:
        user_input_json = input_df.to_json()
        preterm_str = "Pre-term" if preterm_pred else "Normal"
        health_str = "At Risk" if health_pred else "Healthy"

        insert_prediction(cursor, user_input_json, preterm_str, health_str, gestational_period_formatted)
        connection.commit()
        st.success("Prediction saved to the database.")

# Batch Predictions
st.write("## Batch Predictions")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    required_columns = list(manual_data.keys())

    if not all(col in batch_data.columns for col in required_columns):
        st.error("Uploaded file is missing required columns.")
    elif connection:
        # Predict outcomes
        preterm_preds = pipeline_preterm.predict(batch_data[required_columns])
        health_preds = pipeline_health.predict(batch_data[required_columns])

        # Add predictions
        batch_data["Predicted Preterm Birth"] = ["Pre-term" if p == 1 else "Normal" for p in preterm_preds]
        batch_data["Predicted Fetal Health"] = ["At Risk" if p == 1 else "Healthy" for p in health_preds]
        batch_data["Estimated Gestational Period"] = [
            convert_gestational_period(np.random.uniform(28, 36) if p == 1 else np.random.uniform(37, 42)) for p in preterm_preds
        ]

        st.write("### Predictions:")
        st.dataframe(batch_data)

        # Download results
        st.download_button("Download Results", batch_data.to_csv(index=False), "predictions.csv")

        # Add Graphics
        st.write("## Prediction Summary Graphics")

        # Pie chart for Pre-term vs. Normal
        st.write("### Pre-term vs. Normal Distribution")
        preterm_counts = batch_data["Predicted Preterm Birth"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(preterm_counts, labels=preterm_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
        ax.axis('equal')
        st.pyplot(fig)

        # Bar chart for Fetal Health
        st.write("### Fetal Health Distribution")
        health_counts = batch_data["Predicted Fetal Health"].value_counts()
        st.bar_chart(health_counts)

        # Correlation Heatmap
        st.write("### Attribute Correlation Heatmap")
        correlation_matrix = batch_data[required_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        st.pyplot(plt)

# Close the cursor and connection when the app stops
if connection:
    cursor.close()
    connection.close()
