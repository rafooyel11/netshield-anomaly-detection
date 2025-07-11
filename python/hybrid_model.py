import joblib
import json
import numpy as np
import os
from tensorflow import keras

# --- 1. Load All Models and Configuration on Startup ---
def load_all_models():
    """Loads all saved models and config files into memory."""
    print("Loading models and configuration...")

    # get directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
   # get the main project directory
    project_dir = os.path.dirname(script_dir)

    # -- define correct, full paths to saved models and config files
    rf_path = os.path.join(project_dir, 'models', 'random_forest_model.joblib') 
    autoencoder_path = os.path.join(project_dir, 'models', 'autoencoder_model.h5')
    threshold_path = os.path.join(project_dir, 'models', 'anomaly_threshold.txt')
    mapping_path = os.path.join(project_dir, 'json', 'attack_mapping.json')

    # load the files using their paths
    rf_model = joblib.load(rf_path)
    autoencoder_model = keras.models.load_model(autoencoder_path)

    with open(threshold_path, 'r') as f:
        threshold = float(f.read())

    with open(mapping_path, 'r') as f:
        mapping_str_keys = json.load(f)
        attack_mapping = {int(k): v for k, v in mapping_str_keys.items()}

    print("Models and configuration loaded successfully.")
    return rf_model, autoencoder_model, threshold, attack_mapping

# --- 2. The Two-Stage Hybrid Prediction Function ---
def predict_hybrid(data_sample, rf, autoencoder, threshold, mapping):
    """
    Performs a two-stage prediction on a single data sample.
    - data_sample: A single row of preprocessed data (as a NumPy array).
    """
    # Reshape the input data for the model
    data_sample_reshaped = data_sample.reshape(1, -1)

    # --- Stage 1: Autoencoder Anomaly Detection ---
    reconstruction = autoencoder.predict(data_sample_reshaped, verbose=0)
    mse = np.mean(np.square(data_sample_reshaped - reconstruction), axis=1)[0]

    if mse <= threshold:
        return {"status": "Normal", "attack_type": None, "mse": float(mse)}
    
    # --- Stage 2: Random Forest Classification ---
    # The sample is an anomaly, now classify it
    attack_label = rf.predict(data_sample_reshaped)[0]
    attack_name = mapping.get(attack_label, "Unknown Anomaly")
    
    return {"status": "Anomaly", "attack_type": attack_name, "mse": float(mse)}

# --- 3. Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    from load_data import load_and_prepare_data
    
    # Load all the necessary components
    rf_model, autoencoder, threshold, attack_mapping = load_all_models()
    
    # Get test data to make a prediction
    _, _, x_test, y_test = load_and_prepare_data()

    # Test with a normal sample
    normal_sample_index = y_test[y_test == 16].index[0]
    normal_sample = x_test.loc[normal_sample_index].to_numpy()
    prediction = predict_hybrid(normal_sample, rf_model, autoencoder, threshold, attack_mapping)
    print(f"\nPrediction for a NORMAL sample: {prediction}")

    # Test with an attack sample
    attack_sample_index = y_test[y_test != 16].index[0]
    attack_sample = x_test.loc[attack_sample_index].to_numpy()
    prediction = predict_hybrid(attack_sample, rf_model, autoencoder, threshold, attack_mapping)
    print(f"Prediction for an ATTACK sample: {prediction}")