import pandas as pd
import numpy as np
import os
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from indicators import add_indicators

# --- Step 1: Load historical data ---
df = yf.download('TCS.NS', period='6y', interval='1d', auto_adjust=True)

# Clean column names
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
df.columns = df.columns.str.replace(r'\s+', '', regex=True).str.replace('TCS.NS', '').str.strip()

# --- Step 2: Add indicators ---
df = add_indicators(df)

# --- Step 3: Add extra features ---
df['RSI_Lag_1'] = df['RSI'].shift(1)
df['MACD_Change'] = df['MACD'] - df['MACD'].shift(1)
df.dropna(inplace=True)

# --- Step 4: Create labels ---
df['Future_Close'] = df['Close'].shift(-3)
df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']

# Threshold labeling
def label(row):
    if row['Future_Return'] > 0.015:
        return 2  # BUY
    elif row['Future_Return'] < -0.015:
        return 0  # SELL
    else:
        return 1  # HOLD

df['Action'] = df.apply(label, axis=1)

df.dropna(inplace=True)

# --- Step 5: Features and Target ---
features = [
    'Close', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
    'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
    'Bollinger_Upper', 'Bollinger_Lower',
    'Volume_Change', 'Momentum_5d', 'Volatility_7d',
    'Price_Range'
]
X = df[features].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
y = df.loc[X.index, 'Action']
# --- Standardize the features ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# --- Step 6: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)


# --- Step 7: Train model ---
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    eval_metric='mlogloss',
    random_state=42,
    scale_pos_weight=1.5
)

model.fit(X_train, y_train)

# --- Step 8: Evaluate ---
y_pred = model.predict(X_test)
print("\n📊 CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

# --- Optional: Save model ---
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/tcs_model.pkl')
joblib.dump(scaler, '../models/tcs_scaler.pkl')
print("\n✅ Model retrained and saved to: models/tcs_model.pkl")
# Save feature list
joblib.dump(features, '../models/tcs_features.pkl')
print("✅ Features saved to: models/tcs_features.pkl")
