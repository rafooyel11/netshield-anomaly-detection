import pyshark
import requests
import json
import numpy as np
import time

# The URL of your running Flask application's prediction endpoint
FLASK_API_URL = "http://127.0.0.1:5000/predict"

def process_and_predict(packet):
    """
    Extracts features, creates a feature vector,
    and sends it to the Flask API for prediction.
    """
    try:
        # 1. Extract Basic Features from the live packet
        protocol = packet.transport_layer
        src_addr = packet.ip.src
        dst_addr = packet.ip.dst
        length = int(packet.length)

        print(f"Captured: {protocol} packet, Length: {length}, {src_addr} -> {dst_addr}")

        # 2. Create a simplified feature vector for the demo
        feature_vector = np.zeros(119)
        feature_vector[0] = length
        feature_vector[4] = length

        # 3. Send to Flask API for Prediction
        headers = {'Content-Type': 'application/json'}
        data = {"packet_features": feature_vector.tolist(), "protocol": protocol}
        
        requests.post(FLASK_API_URL, headers=headers, data=json.dumps(data), timeout=1)

    except (AttributeError, KeyError):
        # Ignore packets that don't have the required layers
        pass
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the server: {e}")

# --- Start Live Capture ---
print("Starting live network capture on 'Ethernet 2'...")
print("Press Ctrl+C to stop.")

# Use sniff_continuously() which runs indefinitely
capture = pyshark.LiveCapture(interface='Ethernet 2')

for packet in capture.sniff_continuously():
    # This loop will run for each packet that is captured
    process_and_predict(packet)