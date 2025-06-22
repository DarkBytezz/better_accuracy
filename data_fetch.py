import yfinance as yf
import pandas as pd
import os
from indicators import add_indicators

# --- List of Stock Symbols to Fetch ---
symbols = ['TCS.NS']  # Add more like 'INFY.NS', 'RELIANCE.NS' later

# --- Loop through each symbol ---
for symbol in symbols:
    print(f"Fetching data for: {symbol}")

    # Fetch 6 years of daily data
    data = yf.download(symbol, period='6y', interval='1d', auto_adjust=True)

    # Clean and reset index
    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    # Add technical indicators (calling your indicators.py)
    data = add_indicators(data)
    data.dropna(inplace=True)  # Drop rows with NaNs after indicators

    # Save it to a CSV
    os.makedirs('../data', exist_ok=True)
    csv_path = f"../data/{symbol.split('.')[0]}_data.csv"
    data.to_csv(csv_path, index=False)

    print(f"✅ Saved data to {csv_path}\n")
