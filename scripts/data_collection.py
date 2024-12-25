# scripts/data_collection.py

import yfinance as yf
import pandas as pd
import os

def download_crypto_data(symbol='BTC-USD', start='2020-01-01', end='2023-01-01', interval='1d', save_path='data/raw'):
    """
    Downloads historical crypto data from Yahoo Finance.
    """
    data = yf.download(symbol, start=start, end=end, interval=interval)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, f"{symbol}_{interval}.csv")
    data.to_csv(file_path)
    return data

if __name__ == "__main__":
    df = download_crypto_data()
    print(df.head())
