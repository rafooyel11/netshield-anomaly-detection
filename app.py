from flask import Flask, render_template, jsonify, request
import numpy as np
from collections import deque
from datetime import datetime
import os
import sys

# Add the 'python' subfolder to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(script_dir, 'python')
sys.path.insert(0, python_dir)

from python.hybrid_model import load_all_models, predict_hybrid
from python.load_data import load_and_prepare_data

# Initialize Flask app
app = Flask(__name__)

# load all models and configurations 
rf_model, autoencoder, threshold, attack_mapping = load_all_models()

# --- Initialize global variables ---
recent_alerts = deque(maxlen=20)
packet_stats = {"total": 0, "TCP": 0, "UDP": 0, "Other": 0}
# --- FIX: Initialize the missing variable here ---
threats_in_last_interval = 0

# prepare sample data for demo
_, _, x_test, y_test = load_and_prepare_data()
normal_sample = x_test.loc[y_test[y_test == 16].index[0]].to_numpy()
attack_sample = x_test.loc[y_test[y_test != 16].index[0]].to_numpy()
print('Sample data is ready for demo')

# define application routes

@app.route('/')
def dashboard():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    global threats_in_last_interval # Use the global variable
    data = request.get_json()
    
    # Logic for live packet data
    if 'packet_features' in data:
        features = data.get('packet_features')
        protocol = data.get('protocol', 'Other')

        packet_stats["total"] += 1
        if protocol in packet_stats:
            packet_stats[protocol] += 1
        else:
            packet_stats["Other"] += 1
        
        sample_to_predict = np.array(features)
        prediction_result = predict_hybrid(sample_to_predict, rf_model, autoencoder, threshold, attack_mapping)

        if prediction_result['status'] == 'Anomaly':
            prediction_result['timestamp'] = datetime.now().strftime("%H:%M:%S")
            recent_alerts.appendleft(prediction_result)
            threats_in_last_interval += 1 # Increment the counter
        
        return jsonify({"status": "live packet received"}), 200

    # Logic for simulation buttons
    elif 'type' in data:
        data_type = data.get('type')
        sample_to_predict = attack_sample if data_type == 'attack' else normal_sample
        prediction_result = predict_hybrid(sample_to_predict, rf_model, autoencoder, threshold, attack_mapping)
        
        if prediction_result['status'] == 'Anomaly':
            prediction_result['timestamp'] = datetime.now().strftime("%H:%M:%S")
            recent_alerts.appendleft(prediction_result)

        return jsonify(prediction_result)

    return jsonify({"error": "Invalid request format"}), 400


@app.route('/get_dashboard_data')
def get_dashboard_data():
    global threats_in_last_interval # Use the global variable
    data_to_send = {
        "alerts": list(recent_alerts),
        "stats": packet_stats,
        "threats_this_interval": threats_in_last_interval
    }
    # Reset the interval counter after sending it
    threats_in_last_interval = 0
    return jsonify(data_to_send)

# run flask app
if __name__ == '__main__':
    app.run(debug=True)

