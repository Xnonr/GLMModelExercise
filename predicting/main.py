# Imports required libraries
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from fastapi import FastAPI
from fastapi import Request

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

app = FastAPI(title = 'GLM', description = 'GLM API', version = '1.0')

@app.on_event('startup')
def load_model():
    mdl = sm.load('models/glm_final_model.pickle')

@app.post('/predict')
async def get_prediction(info: Request):

    req_info = await info.json()
    return {
        "status": "SUCCESS",
        "data": req_info
    }