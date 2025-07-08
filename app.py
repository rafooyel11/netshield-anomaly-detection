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
    data = request.get_json()
    features = data.get('packet_features')
    protocol = data.get('protocol', 'Other') # Get the protocol from the live capture

    if not features:
        return jsonify({"error": "Missing feature data"}), 400

    # Update packet statistics
    packet_stats["total"] += 1
    if protocol in packet_stats:
        packet_stats[protocol] += 1
    else:
        packet_stats["Other"] += 1

    sample_to_predict = np.array(features)
    prediction_result = predict_hybrid(sample_to_predict, rf_model, autoencoder, threshold, attack_mapping)

    # If an anomaly is detected, add it to our list of recent alerts
    if prediction_result['status'] == 'Anomaly':
        prediction_result['timestamp'] = datetime.now().strftime("%H:%M:%S")
        recent_alerts.appendleft(prediction_result) # Add to the beginning of the list

    return jsonify({"status": "received"}), 200


# An endpoint for the front-end to get new alerts 
@app.route('/get_alerts')
def get_alerts():
    """Provides the list of recent alerts to the dashboard."""
    return jsonify(list(recent_alerts))

# An endpoint to get the dashboard data including recent alerts and stats
@app.route('/get_dashboard_data')
def get_dashboard_data():
    """Provides the list of recent alerts and stats to the dashboard."""
    return jsonify({
        "alerts": list(recent_alerts),
        "stats": packet_stats
    })

# run flask app
if __name__ == '__main__':
    app.run(debug=True)