import pandas as pd
import numpy as np

def add_indicators(df):
    df = df.copy()

    # Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if df.empty:
        raise ValueError("DataFrame is empty. No data was fetched.")

    # --- SMA and Bollinger Bands ---
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    std_20 = df['Close'].rolling(20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + (2 * std_20)
    df['Bollinger_Lower'] = df['SMA_20'] - (2 * std_20)

    # --- EMA ---
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # --- RSI ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']

    # --- Volume Change ---
    df['Volume_Change'] = df['Volume'].pct_change()

    # --- Momentum ---
    df['Momentum_5d'] = df['Close'].pct_change(5)

    # --- Volatility ---
    df['Volatility_7d'] = df['Close'].rolling(7).std()

    # --- Price Range Ratio ---
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

    # --- ADX ---
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    up_move = high.diff()
    down_move = low.diff()

    plus_dm = up_move.copy()
    plus_dm[(up_move <= down_move) | (up_move <= 0)] = 0.0
    minus_dm = down_move.copy()
    minus_dm[(down_move <= up_move) | (down_move <= 0)] = 0.0

    plus_di = 100 * plus_dm.rolling(14).mean() / atr
    minus_di = 100 * minus_dm.rolling(14).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()

    # --- CCI ---
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.fabs(x - x.mean()).mean())
    df['CCI'] = (tp - sma_tp) / (0.015 * mad)

    # --- OBV ---
    close = df['Close'].values
    obv = [0]
    for i in range(1, len(df)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # --- Final Clean-up ---
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("✅ Indicators added. Final DataFrame shape:", df.shape)
    print(df.tail(2).to_string(index=False))

    return df
