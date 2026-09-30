
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
  global model
  model_path = Model.get_model_path('bankrupcy_prediction_model')
  model = joblib.load(model_path)

def run(raw_data):
  try:
    data = json.loads(raw_data)['data'][0]
    data = pd.DataFrame(data)

    result = model.predict(data)

    return json.dumps({"result": result.tolist()})
  except Exception as e:
    return json.dumps({"error": str(e)})
