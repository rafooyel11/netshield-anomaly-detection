import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from load_data import load_and_prepare_data

# --- This script trains and saves everything you need ---

# 1. Load Data
x_train, y_train, x_test, y_test = load_and_prepare_data()
print("\n----------------------------------")


# 2. Train and Save the Random Forest Model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(x_train, y_train)
joblib.dump(rf_model, 'models/random_forest_model.joblib')
print("Random Forest model trained and saved.")
print("\n----------------------------------")


# 3. Train and Save the Autoencoder Model
print("Training Autoencoder model...")
x_train_normal = x_train[y_train == 16]
input_dim = x_train_normal.shape[1]
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train_normal, x_train_normal, epochs=20, batch_size=64, shuffle=True, verbose=0)
autoencoder.save('models/autoencoder_model.h5')
print("Autoencoder model trained and saved.")
print("\n----------------------------------")


# 4. Calculate and Save the Anomaly Threshold
print("Calculating and saving anomaly threshold...")
errors = autoencoder.predict(x_train_normal, verbose=0)
normal_mse = np.mean(np.square(x_train_normal - errors), axis=1)
threshold = np.mean(normal_mse) + 3 * np.std(normal_mse)
with open('models/anomaly_threshold.txt', 'w') as f:
    f.write(str(threshold))
print(f"Anomaly threshold saved.")
print("\n----------------------------------")


# 5. Create and Save the Attack Mapping
print("Creating and saving attack mapping...")
# We need the original text file to create the name-to-number mapping
original_column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]
original_train_df = pd.read_csv('data/KDDTrain+.txt', header=None, names=original_column_names)

le = LabelEncoder()
le.fit(original_train_df['attack_type'])
attack_mapping = {index: label for index, label in enumerate(le.classes_)}

# Convert integer keys to strings for JSON compatibility
attack_mapping_str_keys = {str(k): v for k, v in attack_mapping.items()}

with open('json/attack_mapping.json', 'w') as f:
    json.dump(attack_mapping_str_keys, f)
print("Attack mapping saved.")
print("\n----------------------------------")

print("\nAll components are now saved.")