from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
def home():
    return {"message": "API Running"}

@app.get("/predict")
def predict(years: float):
    result = model.predict(np.array([[years]]))
    return {"salary": result[0]}