#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
import pyowm
from datetime import datetime
import pytz
from models.qas import QAS

lemmatizer = WordNetLemmatizer()
# read model file
model = load_model('models/chatbot_model.h5')
intents = json.loads(open('models/intents.json').read())
words = pickle.load(open('models/words.pkl','rb'))
classes = pickle.load(open('models/classes.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    if len(results) == 0:  # If no result predicted by model
        return_list.append({"intent": "none", "probability": "0"})
    else:
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json,msg):
    tag = ints[0]['intent']
    prob = ints[0]['probability']
    print(prob)
    if float(prob) < 0.85:  # no answer from model, pass to QAS
        tag = "noanswer"
        results = QAS(request.args.get('msg'))
        return results[-1]  # last element contains answer
    if tag == "city":
        place = msg.split(" ")
        city = " ".join(place[1:])
        weather = get_weather(city)
        return weather
    if tag == 'datetime':
        date_time = ''
        tz = pytz.timezone('Asia/Kuala_Lumpur')
        dt = datetime.now(tz)
        date_time += str(dt.strftime("%A")) + ' '
        date_time += str(dt.strftime("%d %B %Y")) + ' '
        date_time += str(dt.strftime("%H:%M:%S"))
        return date_time

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def get_weather(city):
    owm=pyowm.OWM('7481da4f938822e5cf6bf62834c6d5da')
    mgr = owm.weather_manager()
    city_country = f"{city},MY"
    try:
        observation=mgr.weather_at_place(city_country)
        w=observation.weather
        temp=w.temperature('celsius')['temp']
        cloud=w.clouds
        humid=w.humidity
        wind=w.wind()
        city_weather = f"It is approximately {temp} \N{DEGREE SIGN}C in {city.upper()}.<br>Clouds: {cloud}% <br>Humidity: {humid}% <br>Wind Speed:{wind['speed']} km/h"
        return city_weather
    except:
        return "Sorry, please input a valid city [weather (city)]."
    
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents,msg)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

from autocorrect import Speller
spell = Speller()
#print("TESTING speelling:", spell("helo"))

@app.route("/autocorrect")
def autocorrect():
    sentence = request.args.get('inputtxt')
    #print('REACH AUTOCORRECT ', sentence)
    autocorrect_sent= spell(str(sentence))
    if autocorrect_sent != str(sentence):
        return autocorrect_sent
    else:
        return sentence


if __name__ == "__main__":
    app.run()