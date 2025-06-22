
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
from indicators import add_indicators

# --- Step 1: Load trained model ---
model = joblib.load('../models/tcs_model.pkl')
print("✅ Model loaded.")

# --- Step 2: Get latest stock data ---
ticker = 'TCS.NS'
print(f"\n📥 Fetching latest data for {ticker}...")
data = yf.download(ticker, period='90d', interval='1d', auto_adjust=True)

# --- Step 3: Add indicators ---
data = add_indicators(data)
# --- Step 3.5: Clean up messy column names ---
# If MultiIndex: flatten it
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(col).strip() for col in data.columns.values]

# Remove extra spaces and possible ticker suffixes like 'TCS.NS'
data.columns = data.columns.str.replace(r'\s+', '', regex=True)
data.columns = data.columns.str.replace('TCS.NS', '', regex=False)
data.columns = data.columns.str.strip()

# --- Step 4: Rebuild all required features ---
data['RSI_Lag_1'] = data['RSI'].shift(1)
data['MACD_Change'] = data['MACD'] - data['MACD'].shift(1)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Final features (MUST match training exactly)
# Load feature list
features = joblib.load('../models/tcs_features.pkl')


# --- Step 5: Select latest row ---
latest = data[features].tail(1)

if latest.empty:
    print("❌ No valid recent row found.")
    exit()

# --- Step 6: Predict ---
prediction = model.predict(latest)[0]
probs = model.predict_proba(latest)[0]

# --- Step 7: Output ---
reverse_map = {0: "SELL 🚨", 1: "HOLD 🧘‍♂️", 2: "BUY 💸"}

print("\n🔮 AI Decision for Today on TCS:")
print(f"👉 Action: {reverse_map[prediction]}")

print("\n📊 Prediction Confidence:")
for label, prob in zip(model.classes_, probs):
    print(f"{reverse_map[label]}: {prob*100:.2f}%")
