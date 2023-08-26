import pickle
from flask import Flask, request, app,render_template,jsonify,url_for

import numpy as np

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
    print(np.array(list(data.values())).reshape(1,-1)) # data values
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output= model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)










