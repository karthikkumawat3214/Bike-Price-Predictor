from flask import Flask, render_template, request, redirect,jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
cors = CORS(app)
#print("Current working directory:", os.getcwd())
model = pickle.load(open('model.pkl', 'rb'))
bike = pd.read_csv('BikeCleaned.csv')

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['Brand', 'Model', 'Year', 'Seller_Type', 'KM_Driven']),
    remainder='passthrough'
)

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(bike['Brand'].unique())
    bike_models = sorted(bike['Model'].unique())
    year = sorted(bike['Year'].unique(), reverse=True)
    seller_type = bike['Seller_Type'].unique()
    #km_driven = bike['KM_Driven'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, bike_models=bike_models, years=year, seller_types=seller_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('Brand')
        bike_model = request.form.get('Model')
        year = request.form.get('Year')
        seller_type = request.form.get('Seller_Type')
        driven = request.form.get('Kilo_Driven')

        # Include 'Selling_Price' and 'Owner' columns in the prediction DataFrame
        prediction_df = pd.DataFrame(columns=['Brand', 'Model', 'Year', 'Seller_Type', 'KM_Driven', 'Selling_Price', 'Owner'],
                                data=np.array([company, bike_model, year, seller_type, driven, 0, 0]).reshape(1, 7))

        prediction = model.predict(prediction_df)
        print(prediction)

        return str(np.round(prediction[0], 2))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Internal Server Error", 500


if __name__ == '__main__':
    app.run()
