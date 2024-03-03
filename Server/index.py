# index.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import string
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

print("Server started")

# Load the model
model = joblib.load('./model.pkl')

# Load the label encoder
encoder = joblib.load('./label_encoder.pkl')

# Load the vectorizers
with open('./char_vectorizer.pkl', 'rb') as file:
    loaded_char_vectorizer = joblib.load(file)

with open('./word_vectorizer.pkl', 'rb') as file:
    loaded_word_vectorizer = joblib.load(file)

def preprocess(texts):
    clean_text = []
    for text in texts:
        cleaned_text = re.sub(f"[{re.escape(string.punctuation + '–' + string.digits)}]+", " ", text).lower()
        clean_text.append(cleaned_text)
    return clean_text

@app.route('/')
def index():
    return 'Flask server running'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.data.decode('utf-8')
        print('Received features:', features)

        new_data = [features]

        # Preprocess the new data
        cleaned_text = preprocess(new_data)

        # Transform the new data
        new_data_word = loaded_word_vectorizer.transform(cleaned_text).toarray()
        new_data_char = loaded_char_vectorizer.transform(cleaned_text).toarray()

        # Combine character and word features horizontally
        new_data_combined = pd.DataFrame(np.hstack((new_data_word, new_data_char)))

        # Make predictions
        predictions = model.predict(new_data_combined)

        results = [{"text": text, "prediction": encoder.inverse_transform([prediction])[0]} for text, prediction in zip(new_data, predictions)]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run the Flask app on a specified host and port
    app.run(host='0.0.0.0', port=5000, debug=True)
