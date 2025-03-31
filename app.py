<<<<<<< HEAD

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
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, step=1)
temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)

# Convert gender to numerical value (Male=0, Female=1)
gender_val = 0 if gender == "Male" else 1

# Make prediction
if st.button("🔍 Predict Calories"):
    try:
        features = np.array([[gender_val, age, height, heart_rate, temp]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        st.success(f"📊 Predicted Calorie Consumption: {prediction[0]:.2f} kcal")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

=======
# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'calories_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract all form data
    user_inputs = request.form.to_dict()
    # Manually check and remove the desired key-value pair
    for key in list(user_inputs.keys()):
        if key == 'userId' or key == 'duration' or key == 'weight':
            del user_inputs[key]

    # Convert inputs to integers
    int_features = [x for x in user_inputs.values()]
    final_features = [np.array(int_features)]

    # Apply StandardScaler transformation
    final_features_scaled = scaler.transform(final_features)

    # Make prediction
    prediction = model.predict(final_features_scaled)

    return render_template('result.html', 
                           prediction_text=f'Prediction: {prediction} {final_features_scaled}', 
                           user_inputs=user_inputs)

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 23476b6 (initial commit)
