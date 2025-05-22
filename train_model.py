import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib
import tensorflow.keras.backend as K

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ TensorFlow is using GPU!")

# Load dataset
df = pd.read_csv("temperature_sensor_fault_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by="timestamp")

# Normalize features
scaler = StandardScaler()
features = ["temperature", "rate_of_change", "variance"]
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, "feature_scaler.pkl")

# Encode fault labels
label_encoder = LabelEncoder()
df["fault_label"] = label_encoder.fit_transform(df["fault_label"])
joblib.dump(label_encoder, "label_encoder.pkl")

# Convert data into sequences
SEQUENCE_LENGTH = 30

def create_sequences(data, labels, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(labels[i + sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(df[features].values, df["fault_label"].values, SEQUENCE_LENGTH)
X = X.reshape((X.shape[0], X.shape[1], len(features)))
y = to_categorical(y, num_classes=10)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        return K.mean(weight * cross_entropy, axis=-1)
    return loss

# Build BiLSTM Model
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(features))),
    Bidirectional(LSTM(64, activation="tanh", return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32, activation="tanh")),
    Dropout(0.3),
    Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(10, activation="softmax", dtype='float32')
])

# Compile Model
model.compile(
    optimizer=AdamW(learning_rate=0.0005, weight_decay=0.01),
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, mode="min", verbose=1)

# Train model
EPOCHS = 50
BATCH_SIZE = 128

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr, early_stopping],
    shuffle=True
)

# Save model in both .keras and .h5 formats
model.save("bilstm_fault_detection.keras")
model.save("bilstm_fault_detection.h5")
np.save("training_history.npy", history.history)
print("\n✅ Model training complete! Saved as bilstm_fault_detection.keras and bilstm_fault_detection.h5")
print(f"📊 Final Training Accuracy: {history.history['accuracy'][-1]:.5f}")
print(f"📊 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.5f}")
print(f"📉 Final Training Loss: {history.history['loss'][-1]:.5f}")
print(f"📉 Final Validation Loss: {history.history['val_loss'][-1]:.5f}")

# Generate Training Curves Image
import matplotlib.pyplot as plt

# Load history if not already in memory (for standalone use)
history = np.load("training_history.npy", allow_pickle=True).item()

plt.figure(figsize=(12, 8))

# Plot Accuracy
plt.subplot(2, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(2, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Precision
plt.subplot(2, 2, 3)
plt.plot(history['precision'], label='Training Precision')
plt.plot(history['val_precision'], label='Validation Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Plot Recall
plt.subplot(2, 2, 4)
plt.plot(history['recall'], label='Training Recall')
plt.plot(history['val_recall'], label='Validation Recall')
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Training curves image saved as 'training_curves.png'")

# Generate Model Architecture Image
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True, dpi=300)
print("✅ Model architecture image saved as 'model_architecture.png'")