weather_project/
├── app.py
├── models/
│   └── lstm_model.h5
├── data/
│   └── weather_data.csv
├── requirements.txt
└── templates/
    └── index.html


python
Copy code
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load historical weather data
data = pd.read_csv('data/weather_data.csv')

# Load pre-trained LSTM model
model = load_model('models/lstm_model.h5')

# Prepare scaler for features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Temperature', 'Humidity', 'Rainfall', 'Pressure']])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input JSON
    input_data = request.json
    temp = input_data['Temperature']
    hum = input_data['Humidity']
    rain = input_data['Rainfall']
    pres = input_data['Pressure']

    # Scale input
    scaled_input = scaler.transform(np.array([[temp, hum, rain, pres]]))

    # Reshape for LSTM [samples, timesteps, features]
    lstm_input = scaled_input.reshape((1, 1, 4))

    # Predict
    prediction = model.predict(lstm_input)
    # Inverse scale prediction (assuming same scale)
    predicted_values = scaler.inverse_transform(np.hstack([prediction, np.zeros((1,3))]))[:,0]

    return jsonify({'Predicted_Temperature': float(predicted_values[0])})

if __name__ == '__main__':
    app.run(debug=True)
