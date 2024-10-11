from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import logging
import time  # Importing the time module to measure execution time

# Set up logging configuration
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model
model = joblib.load('C:\\Users\\ANANYA ROHATGI\\OneDrive\\Desktop\\phishing url\\phishing_model.pkl')  

app = Flask(__name__)

@app.route('/api-info', methods=['GET'])
def api_info():
    return jsonify({
        "message": "Welcome to the Phishing Detection API",
        "available_endpoints": {
            "/api-info": "Get information about the API",
            "/predict": "Use POST method to get predictions"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()  
    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")

        required_keys = [
            'URLLength',
            'DomainLength',
            'URLSimilarityIndex',
            'CharContinuationRate',
            'URLCharProb',
            'NoOfLettersInURL',
            'LetterRatioInURL',
            'DegitRatioInURL',  
            'NoOfOtherSpecialCharsInURL',
            'SpacialCharRatioInURL',
            'IsHTTPS',
            'LineOfCode',
            'HasTitle',
            'DomainTitleMatchScore',
            'URLTitleMatchScore',
            'HasFavicon',
            'Robots',
            'IsResponsive',
            'HasDescription',
            'NoOfiFrame', 
            'HasSocialNet',
            'HasSubmitButton',
            'NoOfSelfRef',
            'HasHiddenFields',
            'Pay',
            'HasCopyrightInfo',
            'NoOfImage',
            'NoOfJS',
            'NoOfExternalRef',
            'label' 
        ]

       
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing data'}), 400

        
        features = [data[key] for key in required_keys]
        
        prediction = model.predict([features])
        
       
        prediction_value = int(prediction[0]) if isinstance(prediction[0], np.int64) else prediction[0]
        
        execution_time = time.time() - start_time 
        return jsonify({'prediction': prediction_value, 'execution_time': execution_time})

    except Exception as e:
        logging.error("An error occurred: %s", str(e)) 
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
