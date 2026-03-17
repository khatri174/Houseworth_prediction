from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Loading the trained models
model_lr = joblib.load('ML_Models/model.joblib')
scaler = joblib.load('ML_Models/scaler.joblib')

# Function to preprocess input data and make predictions
def predict_price(input_data):
    # Converting input data to DataFrame
    data = pd.DataFrame(input_data, index=[0])

    yes_no_attributes = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']
    non_bin_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    mapping = {'yes': 1, 'no': 0}
    for x in yes_no_attributes:
        data[x] = data[x].map(mapping)

    # Scaling non-binary variables using MinMaxScaler
    data[non_bin_vars] = scaler.transform(data[non_bin_vars])

    # Making prediction
    prediction_lr = model_lr.predict(data)

    return prediction_lr[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    furnishing_status = request.form['furnishing_status']

    if furnishing_status == 'furnished':
        furnishing_semi_furnished = 'no'
        furnishing_unfurnished = 'no'
    elif furnishing_status == 'semi-furnished':
        furnishing_semi_furnished = 'yes'
        furnishing_unfurnished = 'no'
    else:
        furnishing_semi_furnished = 'no'
        furnishing_unfurnished = 'yes'

    # Geting data from the forms
    input_data = {
        'area': int(request.form['area']),
        'bedrooms': int(request.form['bedrooms']),
        'bathrooms': int(request.form['bathrooms']),
        'stories': int(request.form['stories']),
        'mainroad': request.form['mainroad'],
        'guestroom': request.form['guestroom'],
        'basement': request.form['basement'],
        'hotwaterheating': request.form['hotwaterheating'],
        'airconditioning': request.form['airconditioning'],
        'parking': int(request.form['parking']),
        'prefarea': request.form['prefarea'],
        'furnishingstatus_semi-furnished': furnishing_semi_furnished,
        'furnishingstatus_unfurnished': furnishing_unfurnished
    }

    # making prediction
    prediction = round(predict_price(input_data), 2)

    return render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)  