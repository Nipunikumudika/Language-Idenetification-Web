
# print("Started")

import pandas as pd
import re
import string
import joblib
import numpy as np
import sys
import json

features = sys.argv[1]

# Load the saved model
import os

model = joblib.load('./model.pkl')

# Load the saved encoder
encoder = joblib.load('./label_encoder.pkl')

# Load the saved vectorizers
with open('./char_vectorizer.pkl', 'rb') as file:
    loaded_char_vectorizer = joblib.load(file)

with open('./word_vectorizer.pkl', 'rb') as file:
    loaded_word_vectorizer = joblib.load(file)

def preprocess(texts):
    cleanText = []
    for text in texts:
        cleaned_text = re.sub(f"[{re.escape(string.punctuation + '–' + string.digits)}]+", " ", text).lower()
        cleanText.append(cleaned_text)
    return cleanText

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


for text, prediction in zip(new_data, predictions):
    print(json.dumps({"text": text, "prediction": encoder.inverse_transform([prediction])[0]}))



