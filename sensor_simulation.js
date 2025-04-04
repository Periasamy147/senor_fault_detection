// Constants
const MIN_TEMP = 40;  // Matches dataset generation
const MAX_TEMP = 90;
const NORMAL_COUNT = 5;  // 5 normal readings
const FAULT_REPEAT_COUNT = 8;  // 8 readings per fault
const FAULT_TYPES = [
    "stuck_at", "drift", "noise", "out_of_range", 
    "intermittent", "calibration", "failure", "slow_drift", "spike"
];

// Fault tracking variables
let totalReadings = flow.get("totalReadings") || 0;
let phaseCounter = flow.get("phaseCounter") || 0;
let phaseType = flow.get("phaseType") || "normal";
let faultIndex = flow.get("faultIndex") || 0;
let stuckTemperature = flow.get("stuckTemperature") || null;
let driftTemperature = flow.get("driftTemperature") || null;
let intermittentTemperature = flow.get("intermittentTemperature") || null;

// Generate normal sensor data
function generateNormalSensorData() {
    let temperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
    return {
        sensor_id: "DHT22-1",
        temperature: parseFloat(temperature),
        timestamp: new Date().toISOString(),
        fault_type: "normal"
    };
}

// Generate fault data
function generateFaultData(faultType, step) {
    let temperature;
    switch (faultType) {
        case "stuck_at":
            if (stuckTemperature === null) {
                stuckTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
                flow.set("stuckTemperature", stuckTemperature);
            }
            temperature = stuckTemperature;
            break;

        case "drift":
            if (driftTemperature === null) {
                driftTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP);
                flow.set("driftTemperature", driftTemperature);
            }
            driftTemperature += 1.0;
            temperature = Math.min(driftTemperature, MAX_TEMP + 50).toFixed(2);
            break;

        case "noise":
            temperature = (parseFloat(generateNormalSensorData().temperature) + (Math.random() * 10 - 5)).toFixed(2);
            break;

        case "out_of_range":
            temperature = (Math.random() * 10 + 100).toFixed(2);
            break;

        case "intermittent":
            if (intermittentTemperature === null || step % 5 === 0) {
                intermittentTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
                flow.set("intermittentTemperature", intermittentTemperature);
            }
            temperature = intermittentTemperature;
            break;

        case "calibration":
            temperature = (parseFloat(generateNormalSensorData().temperature) + 10).toFixed(2);
            break;

        case "failure":
            temperature = (Math.random() > 0.1) ? generateNormalSensorData().temperature : 0;
            break;

        case "slow_drift":
            if (driftTemperature === null) {
                driftTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP);
                flow.set("driftTemperature", driftTemperature);
            }
            driftTemperature += 0.5;
            temperature = Math.min(driftTemperature, MAX_TEMP + 50).toFixed(2);
            break;

        case "spike":
            temperature = (step === 0) 
                ? (parseFloat(generateNormalSensorData().temperature) + 30).toFixed(2)
                : generateNormalSensorData().temperature;
            break;
    }
    return {
        sensor_id: "DHT22-1",
        temperature: parseFloat(temperature),
        timestamp: new Date().toISOString(),
        fault_type: faultType
    };
}

// Main logic
function generateSensorData() {
    let sensorData;
    node.warn(`Phase: ${phaseType}, Counter: ${phaseCounter}, Fault Index: ${faultIndex}, Total: ${totalReadings}`);

    if (phaseType === "normal") {
        sensorData = generateNormalSensorData();
        phaseCounter++;
        if (phaseCounter >= NORMAL_COUNT) {
            phaseCounter = 0;
            phaseType = FAULT_TYPES[faultIndex];
            node.warn(`Switching to fault: ${phaseType}`);
        }
    } else {
        sensorData = generateFaultData(phaseType, phaseCounter);
        phaseCounter++;
        if (phaseCounter >= FAULT_REPEAT_COUNT) {
            phaseCounter = 0;
            phaseType = "normal";
            faultIndex = (faultIndex + 1) % FAULT_TYPES.length;
            flow.set("stuckTemperature", null);
            flow.set("driftTemperature", null);
            flow.set("intermittentTemperature", null);
            node.warn(`Switching back to normal, next fault index: ${faultIndex}`);
        }
    }

    // Update context
    totalReadings++;
    flow.set("totalReadings", totalReadings);
    flow.set("phaseCounter", phaseCounter);
    flow.set("phaseType", phaseType);
    flow.set("faultIndex", faultIndex);

    // Debug payload explicitly
    node.warn(`Payload: ${JSON.stringify(sensorData)}`);
    return { payload: sensorData };
}

// Execute and send
msg = generateSensorData();
return msg;