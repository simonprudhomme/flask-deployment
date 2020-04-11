import os
import pandas as pd
import dill as pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def main():
    """ Main page of the API """
    return "This is the main page"


@app.route('/predict', methods=['GET'])
def predict():
    """
    API Call
    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')

    except Exception as e:
        raise e

    clf = 'model_v1.pk'

    if test.empty:
        return(bad_request())

    else:
        #Load the saved model
        print("Loading the model...")

        loaded_model = None
        with open(clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test)

        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(list(prediction_series))

        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200
        print("Done")
        return (responses)

if __name__ == '__main__':
    app.run(host='0.0.0.0')