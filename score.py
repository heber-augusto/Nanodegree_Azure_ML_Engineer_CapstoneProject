import pickle
import json
import pandas as pd
import  numpy as  np
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    
    model_path = Model.get_model_path('fraud_detection_model')
    model = joblib.load(model_path)

def run(data):
    try:
        data = pd.DataFrame(json.loads(data)['data'])
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
