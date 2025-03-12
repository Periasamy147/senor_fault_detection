import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time
import tensorflow.keras.backend as K

# AWS DynamoDB setup
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table_name = 'TemperatureReadings'
table = dynamodb.Table(table_name)

# Load trained model and preprocessing objects
try:
    model = load_model(
        'bilstm_fault_detection.keras',
        custom_objects={'loss': lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)},
        compile=False
    )
    scaler = joblib.load('feature_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure model and preprocessing files are in the current directory.")
    exit(1)

# Fault types (matching training)
fault_types = [
    "normal", "stuck_at", "drift", "noise", "out_of_range",
    "intermittent", "calibration", "failure", "slow_drift", "spike"
]

# Sequence length and features
SEQUENCE_LENGTH = 30
FEATURES = ["temperature", "rate_of_change", "variance"]

# Focal Loss (for model compatibility)
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

# Fetch new data from DynamoDB and compute features
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
            print("No new data fetched.")
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
                    'variance': 0  # Will compute in preprocess_data
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
        print(f"Error fetching data from DynamoDB: {e}")
        return []

# Preprocess data with exact training logic
def preprocess_data(buffer):
    if len(buffer) < SEQUENCE_LENGTH:
        print(f"Buffer too small: {len(buffer)} < {SEQUENCE_LENGTH}")
        return None
    
    df = pd.DataFrame(buffer[-SEQUENCE_LENGTH:], columns=['timestamp'] + FEATURES)
    df['variance'] = pd.Series(df['temperature'].values).rolling(window=5, min_periods=1).var().fillna(0)
    X = pd.DataFrame(df[FEATURES], columns=FEATURES)
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((1, SEQUENCE_LENGTH, len(FEATURES)))
    return X_scaled

# Fault detection loop
def detect_faults():
    global data_buffer
    print("data:")
    
    while True:
        new_data = fetch_new_data()
        
        if new_data:
            data_buffer.extend(new_data)
            print(f"Buffer size: {len(data_buffer)}")
            for item in new_data:
                temp = item['temperature']
                output_line = f"{temp}"
                
                if len(data_buffer) >= SEQUENCE_LENGTH:
                    X_input = preprocess_data(data_buffer)
                    if X_input is not None:
                        prediction = model.predict(X_input, verbose=0)
                        predicted_label = np.argmax(prediction, axis=1)[0]
                        fault_type = fault_types[predicted_label]
                        print(f"Predicted fault: {fault_type}, Probabilities: {prediction[0]}")
                        
                        if fault_type == "normal":
                            output_line += " (normal)"
                        elif fault_type == "stuck_at":
                            recent_temps = [d['temperature'] for d in data_buffer[-5:]]
                            if len(set(recent_temps)) == 1:
                                output_line += " (stuck at fault since repeating itself)"
                            else:
                                output_line += " (stuck_at)"
                        elif fault_type == "drift":
                            output_line += " (drift since gradually increasing)"
                        elif fault_type == "noise":
                            output_line += " (noise due to random fluctuations)"
                        elif fault_type == "out_of_range":
                            output_line += " (out of range due to extreme value)"
                        elif fault_type == "intermittent":
                            output_line += " (intermittent since sporadically repeating)"
                        elif fault_type == "calibration":
                            output_line += " (calibration error due to offset)"
                        elif fault_type == "failure":
                            output_line += " (failure due to missing or invalid data)"
                        elif fault_type == "slow_drift":
                            output_line += " (slow drift since slightly increasing)"
                        elif fault_type == "spike":
                            output_line += " (spike due to sudden jump)"
                
                print(output_line)
        
        if len(data_buffer) > 2 * SEQUENCE_LENGTH:
            data_buffer = data_buffer[-2 * SEQUENCE_LENGTH:]
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        detect_faults()
    except KeyboardInterrupt:
        print("Fault detection stopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")