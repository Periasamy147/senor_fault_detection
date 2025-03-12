// Temperature Chart
const tempCtx = document.getElementById('tempChart').getContext('2d');
const tempChart = new Chart(tempCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Temperature (°C)',
            data: [],
            borderColor: '#007BFF',
            backgroundColor: 'rgba(0, 123, 255, 0.3)',
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#007BFF'
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                display: true,
                title: { display: true, text: 'Time', color: '#333', font: { size: 16, family: 'Roboto' } },
                ticks: { color: '#333', font: { size: 12 } }
            },
            y: {
                display: true,
                title: { display: true, text: 'Temperature (°C)', color: '#333', font: { size: 16, family: 'Roboto' } },
                ticks: { color: '#333', font: { size: 12 } }
            }
        },
        plugins: {
            legend: { labels: { color: '#333', font: { size: 14, family: 'Roboto' } } }
        }
    }
});

// Probability Chart
const probCtx = document.getElementById('probChart').getContext('2d');
const probChart = new Chart(probCtx, {
    type: 'bar',
    data: {
        labels: ['Normal', 'Stuck At', 'Drift', 'Noise', 'Out of Range', 'Intermittent', 'Calibration', 'Failure', 'Slow Drift', 'Spike'],
        datasets: [{
            label: 'Probability',
            data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            backgroundColor: '#007BFF',
            borderColor: '#0056b3',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                display: true,
                title: { display: true, text: 'Fault Type', color: '#333', font: { size: 16, family: 'Roboto' } },
                ticks: { color: '#333', font: { size: 12 } }
            },
            y: {
                display: true,
                title: { display: true, text: 'Probability', color: '#333', font: { size: 16, family: 'Roboto' } },
                beginAtZero: true,
                max: 1,
                ticks: { color: '#333', font: { size: 12 } }
            }
        },
        plugins: {
            legend: { labels: { color: '#333', font: { size: 14, family: 'Roboto' } } }
        }
    }
});

// Statistics
let totalReadings = 0;
let faultCount = 0;
let normalCount = 0;
let tempSum = 0;
const faultTypes = ['normal', 'stuck_at', 'drift', 'noise', 'out_of_range', 'intermittent', 'calibration', 'failure', 'slow_drift', 'spike'];

// Update function
function updateDashboard() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            // Update current data
            document.getElementById('currentTemp').textContent = data.temperature.toFixed(2);
            document.getElementById('currentTime').textContent = data.timestamp;
            document.getElementById('currentFault').textContent = data.fault;

            // Update temperature chart
            tempChart.data.labels.push(data.timestamp.split('T')[1].split('.')[0]);
            tempChart.data.datasets[0].data.push(data.temperature);
            if (tempChart.data.labels.length > 60) {
                tempChart.data.labels.shift();
                tempChart.data.datasets[0].data.shift();
            }
            tempChart.update();

            // Update probability chart
            probChart.data.datasets[0].data = data.probabilities;
            probChart.update();

            // Update statistics
            if (data.fault !== "N/A") {
                totalReadings++;
                tempSum += data.temperature;
                if (data.fault === "normal") {
                    normalCount++;
                } else {
                    faultCount++;
                }
            }
            const avgTemp = totalReadings > 0 ? tempSum / totalReadings : 0;
            const faultFreq = totalReadings > 0 ? (faultCount / totalReadings * 100) : 0;
            const maxProb = Math.max(...data.probabilities) * 100;
            const maxFaultIdx = data.probabilities.indexOf(Math.max(...data.probabilities));
            const maxFault = faultTypes[maxFaultIdx];

            document.getElementById('totalReadings').textContent = totalReadings;
            document.getElementById('faultCount').textContent = faultCount;
            document.getElementById('normalCount').textContent = normalCount;
            document.getElementById('avgTemp').textContent = avgTemp.toFixed(2);
            document.getElementById('faultFreq').textContent = faultFreq.toFixed(2);
            document.getElementById('maxProb').textContent = maxProb.toFixed(2);
            document.getElementById('maxFault').textContent = maxFault;
        })
        .catch(error => console.error('Error fetching data:', error));
}

// Fetch data every 5 seconds (changed from 10 seconds)
setInterval(updateDashboard, 5000);
updateDashboard(); // Initial call