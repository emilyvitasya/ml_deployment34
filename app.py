from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title= "Home Credit Prediction",debug=True)



def load_model():
    with open("model_rf_v2.sav", "rb") as file:
        return pickle.load(file)


from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
  return {"message": "API is online!"}


@app.get("/")
def read_root():
  return {"message": "Hello World"}

@app.post("/get_prediction")
def get_prediction(request: list[dict]):
    data=pd.DataFrame(request)
    model=load_model()
    prediction_prob=model.predict_proba(data)[:,1].tolist()
    prediction_class=[1 if i >= 0.5 else 0 for i in prediction_prob]
    return{"class": prediction_class, "prob": prediction_prob}
