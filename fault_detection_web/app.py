import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from flask import Flask, render_template, jsonify
from decimal import Decimal
import tensorflow.keras.backend as K
import logging

# Flask Setup
app = Flask(__name__)

# AWS DynamoDB setup
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
temperature_table = dynamodb.Table('TemperatureReadings')
corrected_table = dynamodb.Table('CorrectedTemperatureReadings')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load model and preprocessing tools
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow(1 - y_pred, gamma)
    return K.mean(weight * cross_entropy, axis=-1)

try:
    model = load_model(
        'bilstm_fault_detection.keras',
        custom_objects={'loss': lambda y_true, y_pred: focal_loss(y_true, y_pred)},
        compile=False
    )
    scaler = joblib.load('feature_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    logger.error(f"Missing file: {e}")
    exit(1)

fault_types = [
    "normal", "stuck_at", "drift", "noise", "out_of_range",
    "intermittent", "calibration", "failure", "slow_drift", "spike"
]

SEQUENCE_LENGTH = 30
FEATURES = ["temperature", "rate_of_change", "variance"]
data_buffer = []
last_timestamp = None
prev_temp = None

def fetch_new_data():
    global last_timestamp, prev_temp
    try:
        if last_timestamp is None:
            response = temperature_table.scan(Limit=SEQUENCE_LENGTH * 2)
        else:
            response = temperature_table.scan(
                FilterExpression="#ts > :t",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={":t": last_timestamp}
            )
        items = sorted(response.get('Items', []), key=lambda x: x['timestamp'])
        new_data = []
        for item in items:
            try:
                timestamp = item['timestamp']
                temperature = float(item.get('temperature', 0))
                sensor_id = item.get('sensor_id', 'tempSensor-01')
                rate_of_change = temperature - prev_temp if prev_temp is not None else 0

                new_data.append({
                    'sensor_id': sensor_id,
                    'timestamp': timestamp,
                    'temperature': temperature,
                    'rate_of_change': rate_of_change,
                    'variance': 0
                })

                if last_timestamp is None or timestamp > last_timestamp:
                    last_timestamp = timestamp
                prev_temp = temperature
            except Exception as e:
                logger.warning(f"Skipping item due to error: {e}")
                continue

        return new_data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return []

def preprocess_data(buffer):
    if len(buffer) < SEQUENCE_LENGTH:
        return None
    df = pd.DataFrame(buffer[-SEQUENCE_LENGTH:], columns=['sensor_id', 'timestamp'] + FEATURES)
    df['variance'] = pd.Series(df['temperature'].values).rolling(window=5, min_periods=1).var().fillna(0)
    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((1, SEQUENCE_LENGTH, len(FEATURES)))
    return X_scaled

def estimate_corrected_temperature(buffer, window=5):
    """Estimate the corrected temperature using recent normal values and slight trend smoothing."""
    normal_temps = [entry['temperature'] for entry in reversed(buffer[:-1]) if entry.get('fault_type', 'normal') == 'normal']
    if len(normal_temps) >= 3:
        avg = np.mean(normal_temps[:window])
        latest = buffer[-2]['temperature'] if len(buffer) > 1 else avg
        # Blend with last known reading to add slight trend
        corrected = round((avg * 0.7 + latest * 0.3), 2)
        return corrected
    elif normal_temps:
        return round(normal_temps[0], 2)
    else:
        return round(buffer[-1]['temperature'], 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def get_data():
    global data_buffer
    new_data = fetch_new_data()
    corrected_data = None

    if new_data:
        data_buffer.extend(new_data)
        if len(data_buffer) > 2 * SEQUENCE_LENGTH:
            data_buffer = data_buffer[-2 * SEQUENCE_LENGTH:]

    latest_data = data_buffer[-1] if data_buffer else {'temperature': 0, 'timestamp': 'N/A', 'sensor_id': 'tempSensor-01'}
    fault_info = {"fault": "N/A", "probabilities": [0.0] * 10}

    if len(data_buffer) >= SEQUENCE_LENGTH:
        X_input = preprocess_data(data_buffer)
        if X_input is not None:
            prediction = model.predict(X_input, verbose=0)
            predicted_label = np.argmax(prediction, axis=1)[0]
            predicted_fault = fault_types[predicted_label]
            fault_info["fault"] = predicted_fault
            fault_info["probabilities"] = prediction[0].tolist()

            # Impute corrected temperature only if faulty
            corrected_temperature = latest_data['temperature']
            if predicted_fault != 'normal':
                corrected_temperature = estimate_corrected_temperature(data_buffer)
                #logger.info(f"Corrected Temperature Imputed: {corrected_temperature}")

            # Store to corrected table
            corrected_data = {
                'sensor_id': latest_data.get('sensor_id', 'tempSensor-01'),
                'timestamp': latest_data['timestamp'],
                'temperature': Decimal(str(corrected_temperature)),
                'fault_type': predicted_fault
            }

            try:
                corrected_table.put_item(Item=corrected_data)
                #logger.info(f"Stored corrected data: {corrected_data}")
            except Exception as e:
                logger.error(f"Failed to write corrected data: {e}")

    return jsonify({
        'temperature': latest_data['temperature'],
        'timestamp': latest_data['timestamp'],
        'fault': fault_info["fault"],
        'probabilities': fault_info["probabilities"]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
