
import streamlit as st
import pickle
import numpy as np
import os

# Load the trained model and scaler

model_path = os.path.join(os.getcwd(), "calories_model.pkl")
scaler_path = os.path.join(os.getcwd(), "scaler.pkl")
# model_path = 'calories_model.pkl'
# scaler_path = 'scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Model or scaler file is missing. Please check the file paths.")
    st.stop()

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("🔥 Calorie Prediction App")
st.write("Enter your details below to predict calorie consumption.")

# Input fields with improved labels and better user experience
age = st.number_input("Age (years)", min_value=1, max_value=100, step=1)
gender = st.radio("Gender", ["Male", "Female"])  # Using radio buttons instead of dropdown
height = st.number_input("Height (cm)", min_value=110, max_value=240, step=1)
Weight = st.number_input("Weight (Kg)", min_value=30, max_value=140, step=1)
Duration = st.number_input("Duration (Min)", min_value=1, max_value=45, step=1)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=55, max_value=140, step=1)
temp = st.number_input("Body Temperature (°C)", min_value=36.5, max_value=43.0, step=0.1)

# Convert gender to numerical value (Male=0, Female=1)
gender_val = 0 if gender == "Male" else 1

# Make prediction
if st.button("🔍 Predict Calories"):
    try:
        features = np.array([[gender_val, age, height,Weight,Duration, heart_rate, temp]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_calories = max(0, prediction[0])
        st.success(f"📊 Predicted Calorie Consumption: {predicted_calories:.2f} kcal")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

