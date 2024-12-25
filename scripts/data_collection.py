from binance.client import Client
import pandas as pd
import os

def download_binance_data(symbol='BTCUSDT', interval='15m', start_str='1 Jan, 2020', save_path='data/raw'):
    """
    Downloads historical crypto data from Binance.
    """
    client = Client()  # You can optionally pass your API key/secret if needed
    klines = client.get_historical_klines(symbol, interval, start_str)
    
    # Convert data to a DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades', 
               'taker_buy_base', 'taker_buy_quote', 'ignore']
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Save to CSV
    interval_save_path = os.path.join(save_path, interval)
    if not os.path.exists(interval_save_path):
        os.makedirs(interval_save_path)
    file_path = os.path.join(interval_save_path, f"{symbol}_{interval}.csv")
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")

    return df

if __name__ == "__main__":
    df = download_binance_data(interval='4h')
    print(df.head())
