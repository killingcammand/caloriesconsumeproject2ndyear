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