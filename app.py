from flask import Flask, render_template, jsonify, request
import numpy as np
from collections import deque
from datetime import datetime

# Import from python directory
from python.hybrid_model import load_all_models, predict_hybrid
from python.load_data import load_and_prepare_data

# Initialize Flask app
app = Flask(__name__)

# load all models and configurations 
rf_model, autoencoder, threshold, attack_mapping = load_all_models()

# Initialize global variables
# This will hold the last 20 alerts for the dashboard
recent_alerts = deque(maxlen=20) # Holds the last 20 alerts
packet_stats = {"total": 0, "TCP": 0, "UDP": 0, "Other": 0}

# prepare sample data for demo
_, _, x_test, y_test = load_and_prepare_data()
normal_sample = x_test.loc[y_test[y_test == 16].index[0]].to_numpy()
attack_sample = x_test.loc[y_test[y_test != 16].index[0]].to_numpy()
print('Sample data is ready for demo')

# create a deque to store the last 20 predictions
recent_alerts = deque(maxlen=20)

# define application routes

@app.route('/')
def dashboard():
    """Render the main dashboard page."""
    return render_template('index.html')

#  The /predict route now handles live data 
@app.route('/predict', methods=['POST'])
def handle_prediction():
    global threats_in_last_interval
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
            threats_in_last_interval += 1
        
        return jsonify({"status": "live packet received"}), 200

    # Logic for simulation buttons
    elif 'type' in data:
        _, _, x_test, y_test = load_and_prepare_data()
        normal_sample = x_test.loc[y_test[y_test == 16].index[0]].to_numpy()
        attack_sample = x_test.loc[y_test[y_test != 16].index[0]].to_numpy()
        
        data_type = data.get('type')
        sample_to_predict = attack_sample if data_type == 'attack' else normal_sample
        prediction_result = predict_hybrid(sample_to_predict, rf_model, autoencoder, threshold, attack_mapping)
        
        if prediction_result['status'] == 'Anomaly':
            prediction_result['timestamp'] = datetime.now().strftime("%H:%M:%S")
            recent_alerts.appendleft(prediction_result)

        return jsonify(prediction_result)

    return jsonify({"error": "Invalid request format"}), 400


# An endpoint for the front-end to get new alerts 
@app.route('/get_alerts')
def get_alerts():
    """Provides the list of recent alerts to the dashboard."""
    return jsonify(list(recent_alerts))

# An endpoint to get the dashboard data including recent alerts and stats
@app.route('/get_dashboard_data')
def get_dashboard_data():
    global threats_in_last_interval
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