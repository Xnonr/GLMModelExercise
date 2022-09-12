#import models.ml.classifier as clf
from fastapi import FastAPI
import statsmodels.api as sm

app = FastAPI(title="GLM API",
              description="API for GLM dataset", version="1.0")

@app.on_event('startup')
def load_model():
    model = sm.load('glm_final_model.pickle')

@app.post('/predict', tags=["predictions"])
async def get_prediction(iris: Iris):
    data = dict(iris)['data']
    prediction = clf.model.predict(data).tolist()
    log_proba = clf.model.predict_log_proba(data).tolist()
    return {"prediction": prediction,
            "log_proba": log_proba}
