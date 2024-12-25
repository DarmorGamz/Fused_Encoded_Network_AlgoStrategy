# scripts/preprocessing.py

import pandas as pd
import numpy as np
import os


def load_and_preprocess(path):
    """
    Loads data from a CSV file, handles missing values, sets index, etc.
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    
    # Ensure 'close' and 'volume' are numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # Drop rows with all NaNs if they exist
    df.dropna(how='all', inplace=True)
    
    # Drop rows where 'volume' is NaN or less than or equal to 0
    df = df[df['volume'] > 0]
    
    # Forward-fill any remaining NaNs
    df.ffill(inplace=True)
    
    return df

def preprocess_data(path='data/raw/BTC-USD_1d.csv', save_path='data/processed'):
    """
    Main preprocessing pipeline.
    """
    df = load_and_preprocess(path)
    # df = add_technical_indicators(df)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    processed_path = os.path.join(save_path, "BTC-USD_1d_processed.csv")
    df.to_csv(processed_path)
    return df

if __name__ == "__main__":
    df_processed = preprocess_data(path="data/raw/BTCUSDT_4h.csv")
    print(df_processed.head())