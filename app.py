from flask import Flask, render_template, jsonify, request
import numpy as np
from collections import deque

# Import from python directory
from python.hybrid_model import load_all_models, predict_hybrid
from python.load_data import load_and_prepare_data

# Initialize Flask app
app = Flask(__name__)

# load all models and configurations 
rf_model, autoencoder, threshold, attack_mapping = load_all_models()

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
    """
    Receives packet data from live_capture.py, runs the model,
    and stores the result if it's an anomaly.
    """
    data = request.get_json()
    features = data.get('packet_features')

    if not features:
        return jsonify({"error": "Missing feature data"}), 400

    sample_to_predict = np.array(features)
    prediction_result = predict_hybrid(sample_to_predict, rf_model, autoencoder, threshold, attack_mapping)

    # If an anomaly is detected, add it to our list of recent alerts
    if prediction_result['status'] == 'Anomaly':
        recent_alerts.append(prediction_result)

    return jsonify({"status": "received"}), 200


# An endpoint for the front-end to get new alerts 
@app.route('/get_alerts')
def get_alerts():
    """Provides the list of recent alerts to the dashboard."""
    return jsonify(list(recent_alerts))

# run flask app
if __name__ == '__main__':
    app.run(debug=True)