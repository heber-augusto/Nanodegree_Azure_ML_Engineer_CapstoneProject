import pickle
import json
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    
    model_path = Model.get_model_path('fraud_detection_model')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()
