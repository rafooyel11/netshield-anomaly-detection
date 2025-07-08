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
    This is the API endpoint. It handles the simple 'type' requests 
    from the test buttons on the dashboard.
    """
    data = request.get_json()
    
    # Check for the 'type' key sent from the front-end JavaScript
    data_type = data.get('type')
    
    if data_type not in ['normal', 'attack']:
        return jsonify({"error": "Invalid 'type' specified in request"}), 400

    # Select which pre-loaded sample to use based on the button clicked
    sample_to_predict = attack_sample if data_type == 'attack' else normal_sample
    
    # Get a prediction from our hybrid model
    prediction_result = predict_hybrid(sample_to_predict, rf_model, autoencoder, threshold, attack_mapping)
    
    # Return the result as a JSON object to the dashboard
    return jsonify(prediction_result)


# An endpoint for the front-end to get new alerts 
@app.route('/get_alerts')
def get_alerts():
    """Provides the list of recent alerts to the dashboard."""
    return jsonify(list(recent_alerts))

# run flask app
if __name__ == '__main__':
    app.run(debug=True)