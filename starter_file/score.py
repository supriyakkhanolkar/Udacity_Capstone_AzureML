import os
import pickle
import json
import joblib

# The entry script has two required functions, init() and run(data). 
# init() is used to initialize the service at startup and load the model  
# run() is used to run the model using request data passed in by a client. 

# Called when the deployed service starts
def init():
    global model

    # Get the path where the deployed model can be found.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './model.pkl')
    # load model
    model = joblib.load(model_path)

{"data": [{
    "id":842517,
    "radius_mean":20.57,
    "texture_mean":17.77,
    "perimeter_mean":132.9,
    "area_mean":1326,
    "smoothness_mean":0.08474,
    "compactness_mean":0.07864,
    "concavity_mean":0.0869,
    "concave points_mean":0.07017,
    "symmetry_mean":0.1812,
    "fractal_dimension_mean":0.05667,
    "radius_se":0.5435,
    "texture_se":0.7339,
    "perimeter_se":3.398,
    "area_se":74.08,
    "smoothness_se":0.005225,
    "compactness_se":0.01308,
    "concavity_se":0.0186,
    "concave points_se":0.0134,
    "symmetry_se":0.01389,
    "fractal_dimension_se":0.003532,
    "radius_worst":24.99,
    "texture_worst":23.41,
    "perimeter_worst":158.8,
    "area_worst":1956,
    "smoothness_worst":0.1238,
    "compactness_worst":0.1866,
    "concavity_worst":0.2416,
    "concave points_worst":0.186,
    "symmetry_worst":0.275,
    "fractal_dimension_worst":0.08902
}]}

# Handle requests to the service
def run(data):
    try:
        data = json.loads(data)
        result = model.predict(data['data'])
        return result.tolist()
    except Exception as e:
        result = str(e)
        return result
