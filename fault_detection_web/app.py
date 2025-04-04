import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from flask import Flask, render_template, jsonify
import time
import tensorflow.keras.backend as K

app = Flask(__name__)

# AWS DynamoDB setup
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table_name = 'TemperatureReadings'
table = dynamodb.Table(table_name)

# Load original trained model and preprocessing objects (for 50,000-sample dataset)
try:
    model = load_model(
        'bilstm_fault_detection.keras',  # Original model
        custom_objects={'loss': lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)},
        compile=False
    )
    scaler = joblib.load('feature_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure model and preprocessing files are in the current directory.")
    exit(1)

# Fault types
fault_types = [
    "normal", "stuck_at", "drift", "noise", "out_of_range",
    "intermittent", "calibration", "failure", "slow_drift", "spike"
]

SEQUENCE_LENGTH = 30
FEATURES = ["temperature", "rate_of_change", "variance"]

# Focal Loss
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow(1 - y_pred, gamma)
    return K.mean(weight * cross_entropy, axis=-1)

# Data buffer
data_buffer = []
last_timestamp = None
prev_temp = None

def fetch_new_data():
    global last_timestamp, prev_temp
    try:
        if last_timestamp is None:
            response = table.scan(Limit=SEQUENCE_LENGTH * 2)
        else:
            response = table.scan(
                FilterExpression="#ts > :t",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={":t": last_timestamp}
            )
        
        items = response.get('Items', [])
        if not items:
            print("No new data fetched (Node-RED now injects every 5 seconds).")
            return []

        items = sorted(items, key=lambda x: x['timestamp'])
        new_data = []
        for item in items:
            try:
                timestamp = item['timestamp']
                temperature = float(item.get('temperature', 0))
                rate_of_change = temperature - prev_temp if prev_temp is not None else 0
                new_data.append({
                    'timestamp': timestamp,
                    'temperature': temperature,
                    'rate_of_change': rate_of_change,
                    'variance': 0  # Computed later
                })
                if last_timestamp is None or timestamp > last_timestamp:
                    last_timestamp = timestamp
                prev_temp = temperature
            except (KeyError, ValueError) as e:
                print(f"Error parsing item: {item}, {e}")
                continue
        
        print(f"Fetched {len(new_data)} new items.")
        return new_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def preprocess_data(buffer):
    if len(buffer) < SEQUENCE_LENGTH:
        return None
    
    df = pd.DataFrame(buffer[-SEQUENCE_LENGTH:], columns=['timestamp'] + FEATURES)
    df['variance'] = pd.Series(df['temperature'].values).rolling(window=5, min_periods=1).var().fillna(0)
    X = pd.DataFrame(df[FEATURES], columns=FEATURES)
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((1, SEQUENCE_LENGTH, len(FEATURES)))
    return X_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def get_data():
    global data_buffer
    new_data = fetch_new_data()
    if new_data:
        data_buffer.extend(new_data)
        print(f"Buffer size: {len(data_buffer)}")
        if len(data_buffer) > 2 * SEQUENCE_LENGTH:
            data_buffer = data_buffer[-2 * SEQUENCE_LENGTH:]

    latest_data = data_buffer[-1] if data_buffer else {'temperature': 0, 'timestamp': 'N/A'}
    fault_info = {"fault": "N/A", "probabilities": [0.0] * 10}
    
    if len(data_buffer) >= SEQUENCE_LENGTH:
        X_input = preprocess_data(data_buffer)
        if X_input is not None:
            prediction = model.predict(X_input, verbose=0)
            predicted_label = np.argmax(prediction, axis=1)[0]
            fault_info["fault"] = fault_types[predicted_label]
            fault_info["probabilities"] = prediction[0].tolist()
            print(f"Predicted: {fault_info['fault']}, Probabilities: {fault_info['probabilities']}")

    response = {
        'temperature': latest_data['temperature'],
        'timestamp': latest_data['timestamp'],
        'fault': fault_info["fault"],
        'probabilities': fault_info["probabilities"]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)