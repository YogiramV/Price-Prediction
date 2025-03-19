from flask import Flask, render_template, request
import pickle
import pandas as pd
from predictor import Predictor

app = Flask(__name__)

# Load the model once
with open('my_variable.pkl', 'rb') as f:
    X_valid = pickle.load(f)

# Mappings for Method and Type
method_map = {
    'S': 'Property Sold',
    'SP': 'Property Sold Prior',
    'PI': 'Property Passed In',
    'PN': 'Sold Prior Not Disclosed',
    'SN': 'Sold Not Disclosed',
    'NB': 'No Bid',
    'VB': 'Vendor Bid',
    'W': 'Withdrawn Prior to Auction',
    'SA': 'Sold After Auction',
    'SS': 'Sold After Auction Price Not Disclosed',
    'N/A': 'Price or Highest Bid Not Available'
}

type_map = {
    'br': 'Bedroom(s)',
    'h': 'House, Cottage, Villa, Semi, Terrace',
    'u': 'Unit, Duplex',
    't': 'Townhouse',
    'dev site': 'Development Site',
    'o res': 'Other Residential'
}


@app.route('/')
def index():
    # List of columns of interest
    columns_of_interest = ['Suburb', 'Type', 'Method', 'Regionname']

    # Extract unique values for the columns of interest
    unique_values = {col: X_valid[col].unique().tolist()
                     for col in columns_of_interest if col in X_valid.columns}

    return render_template(
        'index.html',
        suburb_values=unique_values.get('Suburb', []),
        type_values=unique_values.get('Type', []),
        method_values=unique_values.get('Method', []),
        regionname_values=unique_values.get('Regionname', []),
        type_map=type_map,
        method_map=method_map
    )


@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    suburb = request.form.get('suburb')
    type_ = request.form.get('type')
    method = request.form.get('method')
    regionname = request.form.get('regionname')

    # Additional fields
    rooms = int(request.form.get('rooms'))
    distance = float(request.form.get('distance'))
    postcode = float(request.form.get('postcode'))
    bedroom2 = float(request.form.get('bedroom2'))
    bathroom = float(request.form.get('bathroom'))
    car = float(request.form.get('car'))
    landsize = float(request.form.get('landsize'))
    buildingarea = float(request.form.get('buildingarea'))
    yearbuilt = float(request.form.get('yearbuilt'))
    lattitude = float(request.form.get('lattitude'))
    longtitude = float(request.form.get('longtitude'))
    propertycount = float(request.form.get('propertycount'))
    date = request.form.get('date')

    # Create a dictionary of user inputs
    user_input = {
        'Suburb': suburb,
        'Type': type_,
        'Method': method,
        'Regionname': regionname,
        'Rooms': rooms,
        'Date': date,
        'Distance': distance,
        'Postcode': postcode,
        'Bedroom2': bedroom2,
        'Bathroom': bathroom,
        'Car': car,
        'Landsize': landsize,
        'BuildingArea': buildingarea,
        'YearBuilt': yearbuilt,
        'Lattitude': lattitude,
        'Longtitude': longtitude,
        'Propertycount': propertycount
    }

    # Convert the user input dictionary to a DataFrame
    user_input_df = pd.DataFrame([user_input])

    predictor = Predictor()

    # Assuming 'Predictor' is your trained model
    predicted_price = predictor.predict(user_input_df)

    # Convert abbreviations to full descriptions
    method_description = method_map.get(method, method)
    type_description = type_map.get(type_, type_)

    # Send the full list of options (for the dropdowns) and descriptions to the template
    return render_template(
        'result.html',
        predicted_price=predicted_price[0] if predicted_price else 'N/A',
        suburb=suburb,
        type_=type_description,
        method=method_description,
        date=date,
        regionname=regionname,
        rooms=rooms,
        distance=distance,
        postcode=postcode,
        bedroom2=bedroom2,
        bathroom=bathroom,
        car=car,
        landsize=landsize,
        buildingarea=buildingarea,
        yearbuilt=yearbuilt,
        lattitude=lattitude,
        longtitude=longtitude,
        propertycount=propertycount
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
