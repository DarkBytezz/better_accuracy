from fastapi import FastAPI, Body  # you already have this
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
from indicators import add_indicators

# Load model and features
model = joblib.load('../models/tcs_model.pkl')
features = joblib.load('../models/tcs_features.pkl')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
reverse_map = {0: "SELL 🚨", 1: "HOLD 🧘‍♂️", 2: "BUY 💸"}

@app.get("/")
def root():
    return {"message": "TCS Prediction API 💼🧠"}

@app.post("/predict")
def predict_stock(ticker: str = Body(..., embed=True)):
    data = yf.download(ticker, period='90d', interval='1d', auto_adjust=True)
    data = add_indicators(data)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    data.columns = data.columns.str.replace(r'\s+', '', regex=True)
    data.columns = data.columns.str.replace(ticker, '', regex=False)
    data.columns = data.columns.str.strip()

    data['RSI_Lag_1'] = data['RSI'].shift(1)
    data['MACD_Change'] = data['MACD'] - data['MACD'].shift(1)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    if data.empty:
        return {"error": "Not enough data to make prediction."}

    latest = data[features].tail(1)

    prediction = int(model.predict(latest)[0])
    probs = model.predict_proba(latest)[0]

    result = {
    "prediction": reverse_map[prediction],
    "confidence": {
        reverse_map[int(label)]: float(round(prob * 100, 2))
        for label, prob in zip(model.classes_, probs)
    }
    }

    return result
