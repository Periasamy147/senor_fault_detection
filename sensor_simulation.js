let faultTypes = ["stuck-at", "drift", "noise", "out-of-range", "intermittent", "calibration", "complete failure"];
let state = flow.get("state") || {
    mode: "normal",
    normalCount: 0,
    faultCount: 0,
    faultIndex: 0,
    faultType: "",
    stuckValue: null,
    driftValue: null
};

let temperature;
let fault = "none";

// Generate normal data
if (state.mode === "normal") {
    temperature = parseFloat((Math.random() * (40 - 22) + 22).toFixed(2));
    state.normalCount++;

    // After 10–12 normal readings, switch to fault mode
    if (state.normalCount >= state.normalTargetCount || !state.normalTargetCount) {
        state.mode = "fault";
        state.faultType = faultTypes[state.faultIndex];
        state.faultIndex = (state.faultIndex + 1) % faultTypes.length;
        state.normalCount = 0;
        state.faultCount = 0;
        state.normalTargetCount = Math.floor(Math.random() * 3) + 10; // 10–12
    }

} else if (state.mode === "fault") {
    fault = state.faultType;

    switch (fault) {
        case "stuck-at":
            if (state.stuckValue === null) {
                state.stuckValue = parseFloat((Math.random() * (40 - 22) + 22).toFixed(2));
            }
            temperature = state.stuckValue;
            break;

        case "drift":
            if (state.driftValue === null) {
                state.driftValue = parseFloat((Math.random() * (30 - 25) + 25).toFixed(2));
            }
            temperature = parseFloat((state.driftValue + state.faultCount * 0.3).toFixed(2));
            break;

        case "noise":
            temperature = parseFloat((Math.random() * 20 + 15).toFixed(2));
            break;

        case "out-of-range":
            temperature = parseFloat((Math.random() < 0.5
                ? Math.random() * 10 + 0
                : Math.random() * 30 + 60
            ).toFixed(2));
            break;

        case "intermittent":
            temperature = state.faultCount % 2 === 0
                ? parseFloat((Math.random() * (40 - 22) + 22).toFixed(2))
                : parseFloat((Math.random() * 20 + 10).toFixed(2));
            break;

        case "calibration":
            temperature = parseFloat(((Math.random() * (40 - 22) + 22) + 5).toFixed(2));
            break;

        case "complete failure":
            temperature = 0.0;
            break;
    }

    state.faultCount++;

    // After 7 fault readings, switch back to normal
    if (state.faultCount >= 7) {
        state.mode = "normal";
        state.faultCount = 0;
        state.faultType = "";
        state.stuckValue = null;
        state.driftValue = null;
        state.normalTargetCount = Math.floor(Math.random() * 3) + 10;
    }
}

flow.set("state", state);

// Output to next node
msg.payload = {
    sensor_id: "tempSensor-01",
    timestamp: new Date().toISOString(),
    temperature: temperature,
    fault: fault
};

return msg;
