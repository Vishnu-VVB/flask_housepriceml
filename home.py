from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('./model/real_estate.pickle', 'rb'))

locations = None
data = None

with open('templates\columns.json', 'r') as f:
        data = json.load(f)['data_columns']
        locations = data[5:]


@app.route("/")
def hello():
    global locations
    return render_template('home_prices.html', locations=locations)


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        bhk = int(request.form['bhk'])
        bath = int(request.form['bathrooms'])
        area = float(request.form['area'])
        size = float(request.form['size'])
        status = request.form['status']
        if status=='For Sale':
            status=0
        else:
            status=1

        place = request.form['place']
        print(place)

        global locations
        global data 
        x=np.zeros(len(data))
        x[0]=bhk
        x[1]=bath
        x[2]=area/100
        x[3]=size
        x[4]=status
        place_index = data.index(place.lower())
        x[place_index] = 1

        print(x, '\n' , len(x))

        pred = model.predict([x])[0]


        if len(x) > 0:
            return render_template('home_prices.html', predicted_price = "Predicted price is {}".format(pred))
        else:
            return render_template('home_prices.html', predicted_price = "Invalid")
        
    else:
        return render_template('home_prices.html')

app.run()