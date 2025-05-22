import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Approximate confusion matrix based on precision and test set distribution
# Rows: actual, Columns: predicted (normal, stuck-at, drift, noise)
confusion_matrix = np.array([
    [4900, 40, 30, 30],  # Normal: 5,000 records, 98% precision
    [30, 1920, 30, 20],  # Stuck-at: 2,000 records, 96% precision
    [40, 30, 1395, 35],  # Drift: 1,500 records, 93% precision
    [50, 30, 40, 1380]   # Noise: 1,500 records, 92% precision
])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Stuck-at', 'Drift', 'Noise'],
            yticklabels=['Normal', 'Stuck-at', 'Drift', 'Noise'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Fault Detection (10,000 Test Records)')
plt.savefig('confusion_matrix.png')
plt.close()