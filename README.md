# IndyPublicLibraryChatBot
Project from COMET lab

app.py - code for chatbot implementation
index.html- flask code for user interface code
index.html should be present in a folder named "template"


## Overview: 
This project presents a sophisticated chatbot application designed with Streamlit as the front-end interface, leveraging Natural Language Processing (NLP) through NLTK, and implementing text classification using Scikit-learn's Logistic Regression. The chatbot is pre-configured to handle multiple intents, including responses to common queries and library card-related FAQs, demonstrating scalable and modular architecture for future enhancements.

The primary objective of this chatbot is to provide efficient, intelligent interaction through a robust, vectorized model, offering accurate responses based on user input. The deployment utilizes Streamlit for seamless web interaction, with the underlying machine learning model ensuring accurate predictions across various input patterns.

## Key Features

Text Vectorization with TF-IDF: Input patterns are transformed into meaningful numerical representations using TF-IDF Vectorization, which is a powerful technique for measuring word relevance in the text corpus.
Logistic Regression Model: A trained Logistic Regression classifier predicts the correct intent tag based on input, ensuring precision and scalability as additional intents are incorporated.
Streamlit-Based User Interface: A modern and interactive web UI designed using Streamlit, allowing real-time interaction between users and the chatbot.
Dynamic FAQ Handling: The bot supports dynamic responses with clickable links for specific queries, such as those related to library card services.
Custom SSL Context: NLTK is configured with an SSL context for smooth downloads of necessary tokenization resources.

## System Requirements
Before running the application, ensure that the following dependencies are installed:

- Prerequisites
- Python 3.8+
- Streamlit 1.2+
- NLTK 3.5+
- Scikit-learn 0.24+

## Python Package Installation
You can install the required dependencies using pip:
```sh
pip install streamlit nltk scikit-learn
```
Additionally, NLTK data resources must be downloaded for tokenization:
```sh
import nltk
nltk.download('punkt')
```
# Code Architecture
## 1. Libraries and Dependencies
```sh
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```
The necessary libraries are imported, including NLTK for NLP tasks, Streamlit for the front-end interface, and Scikit-learn for machine learning operations. A secure SSL context is set up to ensure proper downloading of NLTK resources.


## 2. Intent Definitions
```sh
intents = [
    {"tag": "greeting", "patterns": [...], "responses": [...]},
    {"tag": "goodbye", "patterns": [...], "responses": [...]},
    ...
]
```
The intents list contains multiple dictionaries, each representing an intent:

Tag: The identifier for the intent (e.g., "greeting", "goodbye").
Patterns: A collection of user inputs associated with the intent.
Responses: A set of possible responses that the chatbot can choose from when the corresponding intent is identified.
This modular structure facilitates easy addition or modification of intents, making it scalable for future enhancements or specific use cases.

## 3. Text Vectorization and Model Training
The TF-IDF Vectorizer converts textual patterns into feature vectors, while a Logistic Regression classifier is trained on these vectors to predict the intent tags:
```sh
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Extract patterns and tags
patterns = [pattern for intent in intents for pattern in intent['patterns']]
tags = [intent['tag'] for intent in intents for _ in intent['patterns']]

# Training data
x = vectorizer.fit_transform(patterns)
y = tags

# Train the classifier
clf.fit(x, y)
```
The combination of TF-IDF for feature extraction and Logistic Regression ensures robust classification, balancing computational efficiency and prediction accuracy.

## 4. Response Generation
The chatbot's response is determined based on the predicted tag from the user input. The Logistic Regression model classifies the input text, and an appropriate response is selected from the corresponding intent:
```sh
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
```
This function uses the trained classifier to predict the intent tag and retrieve a random response from the corresponding intent. Randomization of responses ensures variability in conversation, leading to a more human-like interaction.

# Streamlit Integration
## Streamlit Interface
The core chatbot functionality is integrated with Streamlit, which serves as the user interface. The interface includes a text input for users to communicate with the bot and renders the bot's response in real-time.
```sh
counter = 0  # To manage input states

def main():
    global counter
    st.title("Chatbot Interface")
    st.write("Welcome to the intelligent chatbot. Type your message below.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.markdown(response)  # Display response as markdown for clickable links
```
**Session Management**: A counter is implemented to ensure each user input is captured separately during the conversation, preventing stream resets in Streamlit.

**User Interaction**: The st.text_input function captures user queries and passes them to the chatbot model for processing. The chatbot's response is then rendered using st.markdown, which supports clickable links where necessary (e.g., library card FAQs).

## Ending Conversations
The chatbot gracefully terminates conversations upon detecting a goodbye message:
```sh
if response.lower() in ['goodbye', 'bye']:
    st.write("Thank you for chatting! Have a great day.")
    st.stop()
```
This feature allows the chatbot to conclude interactions naturally when the user indicates they are done.

# Deployment and Execution
## Running the Application Locally
### 1. Set Up Environment: Ensure the necessary dependencies are installed using the following command:
```sh
pip install streamlit nltk scikit-learn
```
### 2. Launch the Application: Once everything is set up, you can run the Streamlit application via:
```sh
streamlit run <your-script.py>
```
Streamlit will launch a local server and provide a URL that can be accessed through your browser.

# Future Enhancements





# Conclusion
This chatbot application is designed with scalability, maintainability, and ease of deployment in mind. It serves as a solid foundation for developing more advanced conversational agents. With an intuitive interface powered by Streamlit and a machine learning-based back end, this chatbot can be further enhanced to meet specific user needs in various domains.

By using modular design principles and a flexible intent system, the solution is adaptable and easily expandable, making it a powerful tool for both beginners and professionals seeking to deploy intelligent, responsive chatbot systems.


