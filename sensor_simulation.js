// Constants based on dataset
const MIN_TEMP = 45;
const MAX_TEMP = 80;
const NORMAL_COUNT = 15;  // Number of normal readings before faults start
const FAULT_REPEAT_COUNT = 10; // Number of readings for each fault type

// Fault tracking variables (retrieved from context to maintain state)
let readingCount = context.get("readingCount") || 0;
let faultStage = context.get("faultStage") || 0;
let faultCounter = context.get("faultCounter") || 0;
let stuckTemperature = context.get("stuckTemperature") || null;
let driftTemperature = context.get("driftTemperature") || null;
let intermittentTemperature = context.get("intermittentTemperature") || null;

// Function to generate normal sensor data
function generateNormalSensorData() {
    let temperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
    return {
        sensor_id: "DHT22-1",
        temperature: parseFloat(temperature),
        timestamp: new Date().toISOString(),
        fault_type: "normal"
    };
}

// Function to generate faulty sensor data
function generateFaultySensorData() {
    let temperature;
    let faultType = "normal";

    if (faultCounter < NORMAL_COUNT) {
        // Generate normal readings before each fault type
        faultCounter++;
        return generateNormalSensorData();
    } 
    else if (faultCounter < NORMAL_COUNT + FAULT_REPEAT_COUNT) {
        // Introduce faults in a cyclic manner
        switch (faultStage % 5) {
            case 0: // Stuck-at Fault
                if (stuckTemperature === null) {
                    stuckTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
                    context.set("stuckTemperature", stuckTemperature);
                }
                temperature = stuckTemperature;
                faultType = "stuck_at";
                break;

            case 1: // Drift Fault (gradual increase)
                if (driftTemperature === null) {
                    driftTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP);
                    context.set("driftTemperature", driftTemperature);
                }
                driftTemperature += (Math.random() * 1.5 + 0.5); // Gradual increase
                temperature = driftTemperature.toFixed(2);
                faultType = "drift";
                break;

            case 2: // Noise Fault
                temperature = (parseFloat(generateNormalSensorData().temperature) + (Math.random() * 3 - 1.5)).toFixed(2);
                faultType = "noise";
                break;

            case 3: // Out-of-Range Fault
                temperature = (Math.random() * 50 + 120).toFixed(2); // Extreme value
                faultType = "out_of_range";
                break;

            case 4: // Intermittent Fault
                if (intermittentTemperature === null || Math.random() < 0.2) {
                    intermittentTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
                    context.set("intermittentTemperature", intermittentTemperature);
                }
                temperature = intermittentTemperature;
                faultType = "intermittent";
                break;
        }
        faultCounter++;
    } 
    
    if (faultCounter >= NORMAL_COUNT + FAULT_REPEAT_COUNT) {
        faultCounter = 0; // Reset count after each fault type
        faultStage++; // Move to the next fault type
    }
    
    context.set("faultCounter", faultCounter);
    context.set("faultStage", faultStage);

    return {
        sensor_id: "DHT22-1",
        temperature: parseFloat(temperature),
        timestamp: new Date().toISOString(),
        fault_type: faultType
    };
}

// Function to send sensor data every 2 seconds
function sendSensorData() {
    let sensorData = generateFaultySensorData();
    
    node.send([
        { payload: sensorData }  // Send structured fault sequence
    ]);
}

// Execute every 2 seconds (controlled by Inject node)
sendSensorData();

return null;
