from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(filename='loggingfile.log',format="%(levelname)s:%(name)s:%(message)s")
# Load model from pickled file
try:
    with open('C:\\Users\\user\\Downloads\\titanic_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

# Define prediction function
def predict_survival(data):
    # Preprocess input data
    passenger = pd.DataFrame(data, index=[0])
    passenger['Sex'] = passenger['Sex'].map({'male': 0, 'female': 1})

    # Ensure all required features are present
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if not all(feature in passenger.columns for feature in features):
        raise ValueError(f"Missing one of the required features: {features}")

    # Make prediction
    prediction = loaded_model.predict(passenger[features])
    return bool(prediction[0])

# Define route for root URL
@app.route('/', methods=['GET'])
def welcome():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("from the router handler...............................")
    print("we are in prediction")
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            if loaded_model is None:
                raise ValueError("Model is not loaded.")
            survived = predict_survival(data)
            return jsonify({'Survived': survived})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Method not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
