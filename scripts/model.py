# scripts/model.py

import tensorflow as tf
from keras import layers, models
import numpy as np
import pandas as pd
import os

def build_fused_model(input_dim_price=5, input_dim_indicators=6):
    """
    Builds a fused encoding model using Keras functional API.
    
    Args:
        input_dim_price: Number of features in the price input branch (e.g., OHLCV = 5).
        input_dim_indicators: Number of features in the indicators branch.
    Returns:
        A compiled Keras model.
    """
    # Branch A: Price data
    price_input = layers.Input(shape=(input_dim_price,), name="Price_Input")
    x1 = layers.Dense(64, activation='relu')(price_input)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.Dense(32, activation='relu')(x1)
    
    # Branch B: Technical indicators
    indicator_input = layers.Input(shape=(input_dim_indicators,), name="Indicator_Input")
    x2 = layers.Dense(64, activation='relu')(indicator_input)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(32, activation='relu')(x2)
    
    # Optional Branch C: LSTM for time-series
    # Suppose we feed a sequence of N=30 days of raw prices
    seq_input = layers.Input(shape=(30, 5), name="Sequential_Input")  # e.g. 30 time steps, 5 features
    x3 = layers.LSTM(64)(seq_input)
    x3 = layers.Dense(32, activation='relu')(x3)
    
    # Fuse (concatenate) the three branches
    fused = layers.concatenate([x1, x2, x3], axis=-1)
    fused = layers.Dense(64, activation='relu')(fused)
    fused = layers.Dense(32, activation='relu')(fused)
    
    # Output layer: 
    # For classification (e.g. 0 = Sell, 1 = Buy), use 1 neuron with sigmoid
    # For regression (e.g. predict next day price or return), use 1 linear neuron
    output = layers.Dense(1, activation='sigmoid', name="Output")(fused)
    
    model = models.Model(inputs=[price_input, indicator_input, seq_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_training_data(df, window_size=30):
    """
    Prepares data for model training.
    Returns:
        X_price, X_indicators, X_sequence, y
    """
    # Example: We define a binary target. If the next close price is higher than current close => 1 (buy), else 0 (sell).
    # df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df["Target"] = ((df["Close"].shift(-1) > df["Close"]) * 1).fillna(0)
    df.dropna(inplace=True)

    print(df["Target"].unique()) 
    
    # Price input
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X_price = df[price_cols].values
    
    # Indicator input (example: SMA_14, EMA_14, RSI_14, MACD, Signal, BB_Upper, BB_Lower, etc.)
    # Adjust as per the columns you created
    indicator_cols = [col for col in df.columns if col not in price_cols + ['Target']]
    X_indicators = df[indicator_cols].values
    
    # Sequence input: create rolling windows for the last 'window_size' days
    X_sequence = []
    for i in range(len(df) - window_size):
        seq_slice = df.iloc[i:i+window_size][price_cols].values
        X_sequence.append(seq_slice)
    
    # The above approach shortens the dataset for alignment
    # We’ll align the rest of the data
    X_sequence = np.array(X_sequence)
    
    # Because we used (df.iloc[i:i+window_size]), the label for position i+window_size-1
    # is found at i+window_size. So let's align them.
    y = df['Target'].values[window_size:]
    X_price = X_price[window_size:]
    X_indicators = X_indicators[window_size:]
    
    return X_price, X_indicators, X_sequence, y

if __name__ == "__main__":
    # Example usage
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_file = os.path.join(BASE_DIR, 'data', 'processed', 'BTC-USD_1d_processed.csv')
    # processed_file = '../data/processed/BTC-USD_1d_processed.csv'
    df = pd.read_csv(processed_file, parse_dates=True, index_col=0)
    X_price, X_indicators, X_sequence, y = prepare_training_data(df, window_size=30)
    
    # Split into train/test
    split_idx = int(len(X_price) * 0.8)
    X_price_train, X_price_test = X_price[:split_idx], X_price[split_idx:]
    X_ind_train, X_ind_test = X_indicators[:split_idx], X_indicators[split_idx:]
    X_seq_train, X_seq_test = X_sequence[:split_idx], X_sequence[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build model
    model = build_fused_model(
        input_dim_price=X_price_train.shape[1],
        input_dim_indicators=X_ind_train.shape[1],
    )
    
    # Train the model
    model.fit(
        [X_price_train, X_ind_train, X_seq_train], 
        y_train,
        validation_data=([X_price_test, X_ind_test, X_seq_test], y_test),
        epochs=10,
        batch_size=32
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate([X_price_test, X_ind_test, X_seq_test], y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")