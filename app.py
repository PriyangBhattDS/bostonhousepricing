import pickle
from flask import Flask, request, app,render_template,jsonify,url_for

import numpy

import pandas

app = Flask(__name__)

# Load the model

model = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
# POST REQUEST FROM MY SIDE I M GIVING THE INPUT -> THEN MODEL GIVE THE OUTPUT

def predict_api():
    data = request.json['data']
    print(data)










