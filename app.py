from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np
import json


app = Flask(__name__)
model = pickle.load(open("car_price.pkl", "rb"))





@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


# @app.route('/api', methods=['POST'])
# def make_prediction():
#     data = request.get_json(force=True)
#     #convert our json to a numpy array
#     one_hot_data = input_to_one_hot(data)
#     predict_request = gbr.predict([one_hot_data])
#     output = [predict_request[0]]
#     print(data)
#     return jsonify(results=output)

def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(2885)
    # set the numerical input as they are
    enc_input[0] = data['Age_of_car']
    enc_input[1] = data['Mileage']
    ##################### Mark #########################
    # get the array of marks categories
    Brand = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Mercedes-Benz', 'Volkswagen',
       'Ford', 'Mahindra', 'BMW', 'Audi', 'Skoda', 'Tata', 'Renault',
       'Chevrolet', 'Nissan', 'Land', 'Jaguar', 'Mitsubishi', 'Fiat', 'Mini', 
       'Volvo', 'Porsche', 'Jeep', 'Datsun', 'Force', 'ISUZU', 'Bentley',
       'Smart', 'Lamborghini', 'Isuzu', 'Ambassador']
    cols = ['Age_of_car', 'Mileage', 'Fuel_Type', 'Fuel_Type_Diesel',
       'Fuel_Type_CNG', 'Fuel_Type_Petrol', 'Fuel_Type_LPG','Fule_Type_Electric',
       'Brand_Maruti', 'Brand_Hyundai', 'Brand_Honda', 'Brand_Toyota', 'Brand_Mercedes-Benz',
       'Brand_Volkswagen', 'Brand_Ford', 'Brand_Mahindra', 'Brand_BMW',
       'Brand_Audi', 'Brand_Skoda', 'Brand_Tata', 'Brand_Renault',
       'Brand_Chevrolet', 'Brand_Nissan', 'Brand_Land', 'Brand_Jaguar', 'Brand_Mitsubishi',
       'Brand_Fiat', 'Brand_Mini', 'Brand_Volvo', 'Brand_Prosche',
       'Brand_Jeep', 'Brand_Datsum', 'Brand_Force', 'Brand_ISUZU',
       'Brand_Bentley', 'Brand_Smart', 'Brand_Lamborghini', 'Brand_Isuzu', 'Brand_Ambassador']

    # redefine the the user inout to match the column name
    redefinded_user_input = 'Brand_'+data['Brand']
    # search for the index in columns name list 
    mark_column_index = cols.index(redefinded_user_input)
    #print(mark_column_index)
    # fullfill the found index with 1
    enc_input[mark_column_index] = 1
    ##################### Fuel Type ####################
    # get the array of fuel type
    Fuel_Type = ['Diesel', 'Petrol', 'Electric', 'LPG','CNG']
    # redefine the the user inout to match the column name
    redefinded_user_input = 'Fuel_Type_'+data['Fuel_Type']
    # search for the index in columns name list 
    fuelType_column_index = cols.index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[fuelType_column_index] = 1
    return enc_input

@app.route('/api',methods=['POST'])
def get_delay():
    result=request.form
    Age_of_car  = result['Age_of_car']
    Mileage = result['Mileage']
    Brand = result['Brand']
    Fuel_Type = result['Fuel_Type']

    user_input = {'Age_of_car':Age_of_car, 'Mileage':Mileage, 'Fuel_Type':Fuel_Type, 'Brand':Brand}
    
    print(user_input)
    a = input_to_one_hot(user_input)
    price_pred = model.predict([a])[0]
    price_pred = round(price_pred, 2)
    return json.dumps({'price':price_pred});
    # return render_template('result.html',prediction=price_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)







