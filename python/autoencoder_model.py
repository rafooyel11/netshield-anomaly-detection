import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from load_data import load_and_prepare_data

# load data
x_train, y_train, x_test, y_test = load_and_prepare_data()

# check labels in training data
print("Unique labels found in training data:", y_train.unique())

# filter training data to only include normal traffic
x_train_normal = x_train[y_train == 16]

# build the autoencoder model
input_dim = x_train_normal.shape[1]

autoencoder = tf.keras.Sequential([
    # Encoder
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    # Decoder
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])

# compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
print("\nAutoencoder model compiled successfully.")
autoencoder.summary()

# train the autoencoder model
print("\nTraining the autoencoder model...")
history = autoencoder.fit(
    x_train_normal,
    x_train_normal,
    epochs=20,
    batch_size=64,
    shuffle=True,
    validation_split=0.1
)
print("Autoencoder model trained successfully.")

# evaluate the autoencoder model
# calculate the reconstruction error for the entire test set
reconstructions = autoencoder.predict(x_test)
mse = np.mean(np.square(x_test - reconstructions), axis=1)

# set a threshold for anomaly detection
errors = autoencoder.predict(x_train_normal)
normal_mse = np.mean(np.square(x_train_normal - errors), axis=1)
threshold = np.mean(normal_mse) + 3 * np.std(normal_mse)
print(f"Anomaly detection threshold set at: {threshold}")

# classify anomalies
# if reconstruction error > threshold, its an anomaly (1) else normal (0)
y_pred_binary = (mse > threshold).astype(int)

# convert the original multi-class labes to binary labels, normal (0) and anomaly (1)
y_test_binary = (y_test != 16).astype(int)

# display results
print("\nEvaluating model performance...")
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report (0 = Normal, 1 = Anomaly):")
print(classification_report(y_test_binary, y_pred_binary, zero_division=0))