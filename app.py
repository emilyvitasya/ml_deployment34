from fastapi import FastAPI
import pickle
import os
import pandas as pd

app = FastAPI(title="Home Credit Prediction", debug=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_rf_v2.sav")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.get("/")
def read_root():
    return {"message": "API is online!"}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/get_prediction")
def get_prediction(request: list[dict]):
    data = pd.DataFrame(request)
    prediction_prob = model.predict_proba(data)[:, 1].tolist()
    prediction_class = [1 if p >= 0.5 else 0 for p in prediction_prob]
    return {"class": prediction_class, "prob": prediction_prob}
