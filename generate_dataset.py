import pandas as pd
import numpy as np

# Parameters
TOTAL_SAMPLES = 10000
NORMAL_PERCENTAGE = 0.3  # 30% normal data
FAULT_TYPES = ["normal", "stuck_at", "drift", "noise", "out_of_range", "intermittent", "calibration", "failure", "slow_drift", "spike"]
BASE_TEMP = 50.0
TIMESTAMP_START = "2025-03-10T00:00:00"
TEMP_RANGE = (40, 90)

# Fault-specific parameters (matching Node-RED simulation)
NORMAL_COUNT = 10  # 10 normal readings per cycle
STUCK_REPEAT_COUNT = 8  # 8 repeats for stuck_at
DRIFT_STEPS = 10  # 10 steps for drift
NOISE_STD = 5.0
OUT_OF_RANGE_THRESHOLD = 100
INTERMITTENT_INTERVAL = 5
CALIBRATION_OFFSET = 10
FAILURE_DROP_RATE = 0.1
SLOW_DRIFT_RATE = 0.5
SPIKE_MAGNITUDE = 30

def generate_faulty_sensor_data(prev_temp, fault_type, step):
    temp = prev_temp if prev_temp is not None else BASE_TEMP
    if fault_type == "normal":
        temp = np.random.uniform(*TEMP_RANGE)
    elif fault_type == "stuck_at":
        temp = temp  # Remains constant
    elif fault_type == "drift":
        temp += 1.0  # Gradual increase
    elif fault_type == "noise":
        temp += np.random.normal(0, NOISE_STD)
    elif fault_type == "out_of_range":
        temp = OUT_OF_RANGE_THRESHOLD + np.random.uniform(0, 10)
    elif fault_type == "intermittent":
        temp = temp if step % INTERMITTENT_INTERVAL != 0 else np.random.uniform(*TEMP_RANGE)
    elif fault_type == "calibration":
        temp += CALIBRATION_OFFSET
    elif fault_type == "failure":
        temp = temp if np.random.random() > FAILURE_DROP_RATE else 0
    elif fault_type == "slow_drift":
        temp += SLOW_DRIFT_RATE
    elif fault_type == "spike":
        temp += SPIKE_MAGNITUDE if step == 0 else 0
    return max(min(temp, TEMP_RANGE[1] + 50), TEMP_RANGE[0] - 50)  # Extended range for faults

# Generate dataset
data = []
prev_temp = None
timestamps = pd.date_range(start=TIMESTAMP_START, periods=TOTAL_SAMPLES, freq='2s')
window = []

# Simulate Node-RED pattern: 10 normal, 8 stuck_at, 10 drift, etc.
cycle_length = NORMAL_COUNT + STUCK_REPEAT_COUNT + DRIFT_STEPS + 7  # 7 for remaining faults
cycles = TOTAL_SAMPLES // cycle_length

for cycle in range(cycles):
    base_idx = cycle * cycle_length
    
    # Normal (10 readings)
    for i in range(min(NORMAL_COUNT, TOTAL_SAMPLES - base_idx)):
        temp = generate_faulty_sensor_data(prev_temp, "normal", i)
        rate_of_change = temp - prev_temp if prev_temp is not None else 0
        window.append(temp)
        if len(window) > 5: window.pop(0)
        variance = np.var(window) if len(window) > 1 else 0
        data.append({"timestamp": timestamps[base_idx + i], "temperature": temp, "rate_of_change": rate_of_change, "variance": variance, "fault_label": "normal"})
        prev_temp = temp
    
    # Stuck_at (8 repeats)
    stuck_start = base_idx + NORMAL_COUNT
    stuck_temp = prev_temp
    for i in range(min(STUCK_REPEAT_COUNT, TOTAL_SAMPLES - stuck_start)):
        temp = generate_faulty_sensor_data(stuck_temp, "stuck_at", i)
        rate_of_change = temp - prev_temp if prev_temp is not None else 0
        window.append(temp)
        if len(window) > 5: window.pop(0)
        variance = np.var(window) if len(window) > 1 else 0
        data.append({"timestamp": timestamps[stuck_start + i], "temperature": temp, "rate_of_change": rate_of_change, "variance": variance, "fault_label": "stuck_at"})
        prev_temp = temp
    
    # Drift (10 steps)
    drift_start = stuck_start + STUCK_REPEAT_COUNT
    for i in range(min(DRIFT_STEPS, TOTAL_SAMPLES - drift_start)):
        temp = generate_faulty_sensor_data(prev_temp, "drift", i)
        rate_of_change = temp - prev_temp if prev_temp is not None else 0
        window.append(temp)
        if len(window) > 5: window.pop(0)
        variance = np.var(window) if len(window) > 1 else 0
        data.append({"timestamp": timestamps[drift_start + i], "temperature": temp, "rate_of_change": rate_of_change, "variance": variance, "fault_label": "drift"})
        prev_temp = temp
    
    # Other faults (1 each per cycle)
    other_start = drift_start + DRIFT_STEPS
    for j, fault in enumerate(FAULT_TYPES[3:]):  # noise to spike
        if other_start + j >= TOTAL_SAMPLES:
            break
        temp = generate_faulty_sensor_data(prev_temp, fault, 0)
        rate_of_change = temp - prev_temp if prev_temp is not None else 0
        window.append(temp)
        if len(window) > 5: window.pop(0)
        variance = np.var(window) if len(window) > 1 else 0
        data.append({"timestamp": timestamps[other_start + j], "temperature": temp, "rate_of_change": rate_of_change, "variance": variance, "fault_label": fault})
        prev_temp = temp

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("temperature_sensor_fault_dataset.csv", index=False)
print(f"Generated dataset with {len(df)} samples saved to 'temperature_sensor_fault_dataset.csv'.")