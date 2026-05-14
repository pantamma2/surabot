#  Surabot — Bangalore Restaurant Chatbot

A conversational chatbot that helps users discover restaurants in Bangalore based on cuisine preferences and location.

# Preview

BY "Demo.png"
my promt was "I want to eat pizza at Jayanagar" and ouput is "Here are some great restaurants for Pizza cuisine in Jayanagar: LIT Gastro Pub, Enerjuvate Studio & Cafe, Snack N Slurp"

# How It Works

User Input
    |
Bag-of-Words Vectorization (NLTK)
    |
Neural Network (TensorFlow/Keras) & trained on prompts.json
    |
Intent Classification (cuisines / recommendations / greetings / goodbye)
    |
Keyword Extraction (cuisine + location from message)
    |
Zomato CSV Filtering
    |
Response with real restaurant names

# Tech Stack
Python, Flask, NLP, NLTK (tokenization, lemmatization), TensorFlow / Keras (Dense + Dropout), Zomato Bangalore dataset (CSV),  HTML, CSS, JavaScript

# Setup & Installation

1. Clone the repository

git clone https://github.com/your-username/surabot.git
cd surabot

2. Install dependencies

pip install flask tensorflow nltk pandas numpy

3. Download NLTK data

import nltk
nltk.download('punkt')
nltk.download('wordnet')

4. Train the model

python training.py

This generates `words.pkl`, `classes.pkl`, and `chatbotmock1.keras`.

5. Run the app

python chatbot.py

# Dataset

Uses the **Zomato Bangalore Restaurants** dataset containing restaurant names, cuisines, locations, ratings, and more.

# Author

Surya Narayana Murthy
