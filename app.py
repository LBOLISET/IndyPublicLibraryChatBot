import os
import json
import ssl
import random
import nltk
from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL context for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
json_file = "intents_file.json"
with open(json_file, "r") as file:
    intents = json.load(file)

# Preprocess the data
tags = []
patterns = []
responses = {}

for intent in intents:
    pattern_list = intent['Patterns']
    response_list = intent['Responses']
    
    for pattern in pattern_list:
        tags.append(intent['Tag'])
        patterns.append(pattern)
    
    responses[intent['Tag']] = response_list

# Train the model
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(patterns)
y = tags
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(x, y)

# Define the chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    return random.choice(responses.get(tag, ["I'm not sure how to respond to that."]))

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = chatbot(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
