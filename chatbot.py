import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL context for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents including the library card intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    # library card intents 
    {
        "tag": "library_card_faq",
        "patterns": ["library card faq", "what is a library card", "library card information", "details about library card"],
        "responses": ["You can find the frequently asked questions about the library card here: [Library Card FAQ](https://www.indypl.org/get-a-library-card)"]
    },
    {
        "tag": "apply_card_online",
        "patterns": ["apply for library card", "how to apply for a library card", "get a new library card", "sign up for library card"],
        "responses": ["You can apply for a library card online here: [Apply for Library Card](https://register.indypl.org/#/cardSignup)"]
    },
    {
        "tag": "renew_card_online",
        "patterns": ["renew my library card", "library card renewal", "how to renew library card", "renew card online"],
        "responses": ["You can renew your library card online here: [Renew Library Card](https://register.indypl.org/#/cardRenewal)"]
    },
    {
        "tag": "first_library_card",
        "patterns": ["first library card", "getting my first library card", "children's library card", "library card for kids"],
        "responses": ["Information about getting a first library card is available here: [First Library Card](https://www.indypl.org/get-a-library-card/my-first-library-card)"]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define the chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Counter for input tracking
counter = 0

# Streamlit interface for chatbot
def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        
        # Display response with clickable links
        st.markdown(response)
        
        # Stop the conversation if it's a goodbye message
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

# Run the chatbot application
if __name__ == '__main__':
    main()
