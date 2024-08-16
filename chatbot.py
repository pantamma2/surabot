from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

words_data = pickle.load(open('C:\\Users\\chsur\\OneDrive\\Pictures\\Documents\\surabot\\words.pkl', 'rb'))
classes_data = pickle.load(open('C:\\Users\\chsur\\OneDrive\\Pictures\\Documents\\surabot\\classes.pkl', 'rb'))
model_data = tf.keras.models.load_model('C:\\Users\\chsur\\OneDrive\\Pictures\\Documents\\surabot\\chatbotmock1.h5')
prompts_data = json.loads(open('C:\\Users\\chsur\\OneDrive\\Pictures\\Documents\\surabot\\prompts.json').read())
restaurants_data = pd.read_csv('C:\\Users\\chsur\\OneDrive\\Pictures\\Documents\\surabot\\zomato.csv')
context_data = {}

def clean_up_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

def create_bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return(np.array(bag))

def predict(sentence):
    p = create_bow(sentence, words_data, show_details=False)
    res = model_data.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes_data[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents, prompts, restaurants, context):
    tag = intents[0]['intent']
    list_of_intents = prompts['prompts']
    response = ""
    for intent in list_of_intents:
        if intent['tag'] == tag:
            if tag == 'greetings':
                response = random.choice(intent['responses'])
            elif tag == 'goodbye':
                response = random.choice(intent['responses'])
            elif tag == 'cuisines':
                response = random.choice(intent['responses'])
            elif tag == 'recommendations':
                recommended_restaurants = restaurants.sample(3)
                restaurant_names = recommended_restaurants['name'].values.tolist()
                response = f"You could find some good restaurants in Bangalore, here are some: {', '.join(restaurant_names)}"
            break
    return response

@app.route('/')
def index():
    return render_template(r'template\index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json['message']
    intents = predict(message)
    response = get_response(intents, prompts_data, restaurants_data, context_data)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
