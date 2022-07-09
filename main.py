from flask import Flask, request, render_template
import pandas as pd
import pickle
from collections.abc import MutableMapping

app = Flask(__name__)
data=pd.read_csv('/Users/wins/Documents/Web/CHP/house price.csv')
pipe=pickle.load(open("/Users/wins/Documents/Web/CHP/model.pkl", 'rb'))

@app.route('/')
def index():
    locations= sorted(data['location'].unique())
    return render_template('app.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    print(location, bhk, bath, total_sqft)
    input=pd.DataFrame([[location,total_sqft,bath,bhk]], columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0] *1e5

    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True)
