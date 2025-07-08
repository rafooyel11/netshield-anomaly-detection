from flask import Flask, render_template, jsonify, request
import numpy as np

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

# define application routes

@app.route('/')
def dashboard():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """ API endpoint to handle prediction requests."""
    data = request.get_json()
    sample_to_predict = attack_sample if data.get('attack') else normal_sample
    prediction_result = predict_hybrid(sample_to_predict, rf_model, autoencoder, threshold, attack_mapping)
    return jsonify(prediction_result)

# run flask app
if __name__ == '__main__':
    app.run(debug=True)