# scripts/preprocessing.py

import pandas as pd
import numpy as np
import os


def load_and_preprocess(path):
    """
    Loads data from a CSV file, handles missing values, sets index, etc.
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    
    # Ensure 'Close' and 'Volume' are numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Drop rows with all NaNs if they exist
    df.dropna(how='all', inplace=True)
    
    # Drop rows where 'Volume' is NaN or less than or equal to 0
    df = df[df['Volume'] > 0]
    
    # Forward-fill any remaining NaNs
    df.ffill(inplace=True)
    
    return df

def add_technical_indicators(df):
    # Simple Moving Average
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    
    # Exponential Moving Average
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands (20 period)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + 2*std20
    df['BB_Lower'] = sma20 - 2*std20
    
    df.dropna(inplace=True)
    return df

def preprocess_data(path='data/raw/BTC-USD_1d.csv', save_path='data/processed'):
    """
    Main preprocessing pipeline.
    """
    df = load_and_preprocess(path)
    df = add_technical_indicators(df)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    processed_path = os.path.join(save_path, "BTC-USD_1d_processed.csv")
    df.to_csv(processed_path)
    return df

if __name__ == "__main__":
    df_processed = preprocess_data()
    print(df_processed.head())