import pyshark
import requests
import json
import numpy as np
import time

# The URL of flask application 
FLASK_API_URL = "http://127.0.0.1:5000/predict"

def process_and_predict(packet):
    
    try:
        # extract features from the packet
        protocol = packet.transport_layer 
        src_addr = packet.ip.src
        dst_addr = packet.ip.dst
        length = int(packet.length)

        print(f"Captured: {protocol} packet, Length: {length}, {src_addr} -> {dst_addr}")

        # create feature vector that matches model's input
        feature_vector = np.zeros(119)
        feature_vector[0] = length 
        feature_vector[4] = length

        # send to flask api for prediction
        headers = {'Content-Type': 'application/json'}
        # convert the numpy array to a list for JSON serialization
        data = {"packet_features": feature_vector.tolist()}

    except (AttributeError, KeyError):
        pass

    # start live capture
    print("Starting live network capture and sending to netshield server...")
    capture = pyshark.LiveCapture(interface='Wi-Fi')

    # apply the preprocessing fucntion to each captured packet
    capture.apply_on_packets(process_and_predict)
