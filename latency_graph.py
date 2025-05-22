import numpy as np
import matplotlib.pyplot as plt

# Simulate latency data: 1,000 samples, mostly 5–6 ms, max 7 ms
np.random.seed(42)
latency_data = np.random.normal(loc=5.5, scale=0.5, size=1000)  # Mean 5.5 ms, std 0.5 ms
latency_data = np.clip(latency_data, 5.0, 7.0)  # Clip to 5–7 ms range

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(latency_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=6.0, color='red', linestyle='--', label='90th Percentile (6 ms)')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.title('Inference Latency Distribution (1,000 Real-Time Samples)')
plt.legend()
plt.savefig('latency_graph.png')
plt.close()