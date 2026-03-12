# student_score_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load trained model and scaler
# -----------------------------
BASE_DIR = os.path.dirname(__file__)  # ensures correct path on Streamlit Cloud
model_path = os.path.join(BASE_DIR, "student_score_model_best.pkl")
scaler_path = os.path.join(BASE_DIR, "student_score_scaler_best.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Student Exam Score Prediction")
st.header("Enter student details:")

# Numeric inputs
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=100, value=10)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)

# Feature engineering
study_efficiency = hours_studied / (sleep_hours + 0.1)

# Motivation level
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
motivation_map = {"Low": 0, "Medium": 1, "High": 2}
motivation_score = motivation_map[motivation_level]

# Categorical inputs
categorical_cols = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender"
]

categorical_inputs = {}
for col in categorical_cols:
    options = st.selectbox(col, ["Low", "Medium", "High", "Yes", "No", "Public", "Private", "Near", "Far", "Unknown", "Male", "Female"])
    categorical_inputs[col] = options

# Encode categorical values
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(["Low", "Medium", "High", "Yes", "No", "Public", "Private", "Near", "Far", "Unknown", "Male", "Female"])
    categorical_inputs[col] = le.transform([categorical_inputs[col]])[0]

# -----------------------------
# Create input dataframe
# -----------------------------
input_dict = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Study_Efficiency": study_efficiency,
    "Motivation_Level_Score": motivation_score
}

# Add categorical features
for col in categorical_cols:
    input_dict[col] = categorical_inputs[col]

input_df = pd.DataFrame([input_dict])

# Scale numeric features
numeric_features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
                    "Study_Efficiency", "Motivation_Level_Score"]
input_df[numeric_features] = scaler.transform(input_df[numeric_features])

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Exam Score"):
    pred_score = model.predict(input_df)[0]
    st.success(f"Predicted Exam Score: {pred_score:.2f}")
