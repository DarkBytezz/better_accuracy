# 🔮 Stock Prediction API (FastAPI)

This is a machine learning API that takes a stock ticker and returns a prediction: BUY 💸, HOLD 🧘‍♂️, or SELL 🚨, with confidence percentages.

The model and preprocessing are already handled. You just need to run this API and hit the /predict endpoint from your frontend/backend.

---

## 🚀 Getting Started

1. Clone the Repo
git clone https://github.com/<your-repo-link>.git
cd <your-repo>

2. Install Requirements
pip install -r requirements.txt

3. Run the API
uvicorn main:app --reload

The server will start at:
http://127.0.0.1:8000

API docs available at:
http://127.0.0.1:8000/docs

---

## 🔌 API Endpoint

POST /predict

✅ Request Body (JSON)
{
  "ticker": "TCS.NS"
}

Replace "TCS.NS" with any valid stock ticker (Yahoo Finance format)

---

📤 Response
{
  "prediction": "HOLD 🧘‍♂️",
  "confidence": {
    "SELL 🚨": 18.73,
    "HOLD 🧘‍♂️": 75.66,
    "BUY 💸": 5.61
  }
}

---

## 🔗 How to Use in Website / Frontend

- Your backend or frontend just needs to:
  - Send a POST request to http://localhost:8000/predict (or deployed URL).
  - Include a JSON body with the stock ticker.
  - Parse the JSON response and display the prediction + confidence.

---

## 🧠 Behind the Scenes

No need to worry about these unless you're modifying logic:

- main.py – FastAPI app with routes
- indicators.py – adds RSI, MACD, etc.
- tcs_model.pkl, tcs_features.pkl – pre-trained model and assets

---

## ✅ Final Notes

- You only need main.py running. All other logic is auto-used.
- CORS is enabled, so frontend calls will work directly.
- Ready for integration in React/Next.js/Node.js/anything.
