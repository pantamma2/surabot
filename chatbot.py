from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import nltk
import os
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)  # Fixed: removed trailing comma

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# Fixed: use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

words_data    = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
classes_data  = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))
model_data    = tf.keras.models.load_model(os.path.join(BASE_DIR, 'chatbotmock1.keras'))
prompts_data  = json.load(open(os.path.join(BASE_DIR, 'prompts.json')))
restaurants_data = pd.read_csv(os.path.join(BASE_DIR, 'zomato.csv'), encoding='cp1252')

# Normalize column names to be safe
restaurants_data.columns = restaurants_data.columns.str.strip().str.lower()

context_data = {}

def clean_up_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in words]

def create_bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict(sentence):
    p = create_bow(sentence, words_data)
    res = model_data.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes_data[r[0]], "probability": str(r[1])} for r in results]

def get_restaurants_by_cuisine(restaurants, cuisine_keyword):
    """Filter restaurants by cuisine keyword."""
    if 'cuisines' in restaurants.columns:
        filtered = restaurants[
            restaurants['cuisines'].str.contains(cuisine_keyword, case=False, na=False)
        ]
        if not filtered.empty:
            sample = filtered.sample(min(3, len(filtered)))
            return sample['name'].values.tolist()
    # Fallback: return random restaurants
    return restaurants.sample(min(3, len(restaurants)))['name'].values.tolist()

# Detect keywords from user message
CUISINE_KEYWORDS = [
    'italian', 'chinese', 'indian', 'thai', 'mexican', 'japanese',
    'continental', 'north indian', 'south indian', 'pizza', 'biryani',
    'burger', 'sushi', 'kebab', 'dosa', 'udupi'
]

LOCATION_KEYWORDS = [
    'koramangala', 'indiranagar', 'hsr', 'jayanagar', 'whitefield',
    'mg road', 'brigade road', 'btm', 'jp nagar', 'marathahalli',
    'electronic city', 'yelahanka', 'hebbal', 'bannerghatta'
]

def extract_keywords(message):
    msg = message.lower()
    detected_cuisine  = next((c for c in CUISINE_KEYWORDS  if c in msg), None)
    detected_location = next((l for l in LOCATION_KEYWORDS if l in msg), None)
    return detected_cuisine, detected_location

def filter_restaurants(restaurants, cuisine=None, location=None):
    df = restaurants.copy()

    if cuisine and 'cuisines' in df.columns:
        df = df[df['cuisines'].str.contains(cuisine, case=False, na=False)]

    if location and 'location' in df.columns:
        df = df[df['location'].str.contains(location, case=False, na=False)]

    if df.empty:
        # Fallback: relax one filter at a time
        df = restaurants.copy()
        if cuisine and 'cuisines' in df.columns:
            df = df[df['cuisines'].str.contains(cuisine, case=False, na=False)]
        if df.empty:
            df = restaurants  # final fallback: random

    return df.sample(min(3, len(df)))['name'].values.tolist()

def get_response(intents, prompts, restaurants, context):
    if not intents:
        return "I didn't quite catch that. Try asking about restaurants or cuisines in Bangalore!"

    tag = intents[0]['intent']
    message = context.get('last_message', '')
    cuisine, location = extract_keywords(message)

    # Build a human-readable context string for the response
    context_parts = []
    if cuisine:  context_parts.append(f"{cuisine.title()} cuisine")
    if location: context_parts.append(f"{location.title()}")
    context_str = " in ".join(context_parts) if context_parts else "Bangalore"

    for intent in prompts['prompts']:
        if intent['tag'] == tag:

            if tag in ('greetings', 'goodbye', 'thanks'):
                return random.choice(intent['responses'])

            elif tag in ('cuisines', 'recommendations', 'location'):
                names = filter_restaurants(restaurants, cuisine, location)
                return f"Here are some great restaurants for {context_str}: {', '.join(names)}"

            else:
                return random.choice(intent['responses'])

    return "Try asking me about restaurants, cuisines, or areas in Bangalore!"

@app.route('/')
def index():
    return render_template('index.html')  # Fixed: proper template path

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    if not data or 'message' not in data:
        return jsonify("Please send a message!"), 400

    message = data['message']
    context_data['last_message'] = message  # Store for cuisine detection

    intents = predict(message)
    response = get_response(intents, prompts_data, restaurants_data, context_data)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
