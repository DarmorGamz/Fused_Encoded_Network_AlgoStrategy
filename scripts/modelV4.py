# fen_model.py

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

###############################################################################
# A. ADVANCED TECHNICAL INDICATORS
###############################################################################

def compute_rsi(series, window=14):
    """
    Standard RSI (Wilder's) in [0, 100].
    """
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.ewm(span=window, adjust=False).mean()
    avg_loss = losses.ewm(span=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_stoch_rsi(close, high, low, window=14, smoothK=3, smoothD=3):
    """
    Stochastic RSI: 
      stoch_rsi_k = SMA( (RSI - min(RSI)) / (max(RSI) - min(RSI)) * 100 ) 
      stoch_rsi_d = SMA(stoch_rsi_k)
    """
    rsi = compute_rsi(close, window=window)
    min_rsi = rsi.rolling(window).min()
    max_rsi = rsi.rolling(window).max()
    stoch_rsi_k = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9) * 100
    stoch_rsi_k_smoothed = stoch_rsi_k.rolling(smoothK).mean()
    stoch_rsi_d_smoothed = stoch_rsi_k_smoothed.rolling(smoothD).mean()
    return stoch_rsi_k_smoothed, stoch_rsi_d_smoothed

def compute_adx(high, low, close, window=14):
    """
    Average Directional Index (ADX) local implementation.
    ADX indicates trend strength in [0,100].
    """
    prev_close = close.shift(1)
    high_low = high - low
    high_pc = (high - prev_close).abs()
    low_pc = (low - prev_close).abs()
    tr = high_low.combine(high_pc, max).combine(low_pc, max)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / window
    tr_smoothed = pd.Series(tr).ewm(alpha=alpha).mean()
    plus_dm_smoothed = pd.Series(plus_dm).ewm(alpha=alpha).mean()
    minus_dm_smoothed = pd.Series(minus_dm).ewm(alpha=alpha).mean()

    plus_di = 100.0 * plus_dm_smoothed / (tr_smoothed + 1e-9)
    minus_di = 100.0 * minus_dm_smoothed / (tr_smoothed + 1e-9)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100.0
    adx = dx.ewm(alpha=alpha).mean()
    return adx

def compute_obv(close, volume):
    """
    On-Balance Volume (OBV):
      - Add volume on an up day, subtract on a down day.
    """
    diff = close.diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    return (direction * volume).cumsum()

def compute_atr(high, low, close, window=14):
    """
    Average True Range (ATR), basic version.
    """
    prev_close = close.shift(1)
    high_low = high - low
    high_close = (high - prev_close).abs()
    low_close = (low - prev_close).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window).mean()
    return atr

def add_advanced_indicators(df):
    """
    Adds a set of advanced indicators to the DataFrame in-place:
      - Stoch RSI (K, D)
      - ADX
      - OBV
      - ATR
    Adjust or add more as needed.
    """
    # Stoch RSI
    df['StochRSI_K'], df['StochRSI_D'] = compute_stoch_rsi(
        df['close'], df['high'], df['low'], window=14, smoothK=3, smoothD=3
    )
    # ADX
    df['ADX'] = compute_adx(df['high'], df['low'], df['close'], window=14)
    # OBV
    df['OBV'] = compute_obv(df['close'], df['volume'])
    # ATR
    df['ATR_14'] = compute_atr(df['high'], df['low'], df['close'], window=14)

    df.dropna(inplace=True)
    return df

###############################################################################
# B. DATA PREPARATION & LABELING
###############################################################################

def create_multiday_label(df, horizon=5):
    """
    Label = 1 if close[t + horizon] > close[t], else 0.
    """
    df['Target'] = (df['close'].shift(-horizon) > df['close']).astype(int)

def scale_features(df, cols):
    """
    Simple standard scaling of selected columns in-place.
    """
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def prepare_data_for_fen(df, price_cols, indicator_cols, horizon=5, window_size=30):
    """
    Build the arrays for FEN:
      - X_price: single-time price features
      - X_ind: single-time indicator features
      - X_seq: time-series (LSTM input) from price
      - y: multi-day horizon label
      - close_prices: for potential backtesting
    """
    # 1) Create multi-day horizon label
    create_multiday_label(df, horizon)
    df.dropna(inplace=True)

    # 2) Scale relevant columns
    all_cols = list(set(price_cols + indicator_cols))
    df = scale_features(df, all_cols)

    # 3) Build arrays
    price_array = df[price_cols].values
    indicator_array = df[indicator_cols].values
    close_prices = df['close'].values

    # Build sequence data (LSTM) from price_array
    seq_data = []
    for i in range(len(df) - window_size):
        seq_data.append(price_array[i : i + window_size])
    seq_data = np.array(seq_data)

    # Align everything
    y = df['Target'].values[window_size:]
    close_prices = close_prices[window_size:]
    price_array = price_array[window_size:]
    indicator_array = indicator_array[window_size:]

    return price_array, indicator_array, seq_data, y, close_prices

###############################################################################
# C. FUSED ENCODED NETWORK MODEL
###############################################################################

def build_fen_model(
    input_dim_price,
    input_dim_indicators,
    seq_length,
    seq_features,
    hidden_units=64,
    l2_reg=0.001,
    dropout_rate=0.2
):
    """
    Fused Encoded Network:
      - Branch A: Price features (dense)
      - Branch B: Indicators (dense)
      - Branch C: LSTM + MultiHeadAttention on sequential data
      - Fusion -> Dense -> Sigmoid (binary classification)
    """

    # Branch A: Price
    price_input = layers.Input(shape=(input_dim_price,), name='Price_Input')
    x_price = layers.Dense(hidden_units, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(price_input)
    x_price = layers.BatchNormalization()(x_price)
    x_price = layers.Dropout(dropout_rate)(x_price)
    x_price = layers.Dense(hidden_units//2, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(x_price)
    x_price = layers.BatchNormalization()(x_price)

    # Branch B: Indicators
    indicator_input = layers.Input(shape=(input_dim_indicators,), name='Indicator_Input')
    x_ind = layers.Dense(hidden_units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(indicator_input)
    x_ind = layers.BatchNormalization()(x_ind)
    x_ind = layers.Dropout(dropout_rate)(x_ind)
    x_ind = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_ind)
    x_ind = layers.BatchNormalization()(x_ind)

    # Branch C: LSTM + MultiHeadAttention
    seq_input = layers.Input(shape=(seq_length, seq_features), name='Sequential_Input')
    x_seq = layers.LSTM(hidden_units, return_sequences=True,
                        kernel_regularizer=regularizers.l2(l2_reg))(seq_input)
    x_seq = layers.BatchNormalization()(x_seq)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=hidden_units//4)(x_seq, x_seq)
    x_seq = layers.Add()([x_seq, attn])  # Residual
    x_seq = layers.LayerNormalization()(x_seq)
    x_seq = layers.GlobalAveragePooling1D()(x_seq)
    x_seq = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_seq)
    x_seq = layers.BatchNormalization()(x_seq)

    # Fuse
    fused = layers.Concatenate(axis=-1)([x_price, x_ind, x_seq])
    fused = layers.Dense(hidden_units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(dropout_rate)(fused)
    fused = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)

    # Output
    output = layers.Dense(1, activation='sigmoid', name='Output')(fused)
    model = models.Model(inputs=[price_input, indicator_input, seq_input], outputs=output)

    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

###############################################################################
# D. SIMPLE BACKTESTING & SHARPE CALCULATION
###############################################################################

def simple_backtest(predictions, close_prices, horizon=1, cost=0.0005):
    """
    Basic daily backtest:
      - If prediction > 0.5 => go long next day; else go short.
      - Subtract cost each position flip (assume daily for simplicity).
      - horizon=1 => new signal daily. If horizon>1, adapt logic if you hold multiple days.

    Returns:
      daily_returns (pd.Series), cumrets (pd.Series), win_rate, final_cum_ret, sharpe
    """
    # Convert predictions to +/- 1
    signal = np.where(predictions > 0.5, 1.0, -1.0)

    # Next-day returns (simple approach): ret[t] = (Close[t+1] - Close[t]) / Close[t]
    # We shift close by 1 to get next day. 
    shifted_close = pd.Series(close_prices).shift(-horizon)
    current_close = pd.Series(close_prices)
    ret = (shifted_close - current_close) / (current_close + 1e-9)

    # Strategy return
    strat_ret = signal * ret.values

    # Subtract cost daily (assuming we open/close each day)
    strat_ret -= cost

    # Convert to pd.Series
    daily_returns = pd.Series(strat_ret, index=current_close.index)
    daily_returns.dropna(inplace=True)

    # Win rate
    wins = daily_returns[daily_returns > 0].count()
    total_trades = daily_returns.count()
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Cumulative returns
    cumrets = (1.0 + daily_returns).cumprod()
    final_cum_ret = cumrets.iloc[-1] - 1.0

    # Sharpe (annualized)
    sharpe_val = annualized_sharpe(daily_returns, trading_days=365)  # or 252 if equities

    return daily_returns, cumrets, win_rate, final_cum_ret, sharpe_val

def annualized_sharpe(daily_returns, trading_days=252):
    """
    Annualized Sharpe Ratio = mean(daily_returns)*trading_days / (std(daily_returns)*sqrt(trading_days)).
    """
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    if std_ret < 1e-9:
        return 0.0
    return (mean_ret * trading_days) / (std_ret * np.sqrt(trading_days))

###############################################################################
# E. EXAMPLE MAIN
###############################################################################

if __name__ == "__main__":
    """
    Example usage of the FEN model with advanced indicators.
    1) Load a CSV with columns like: 
       [timestamp, open, high, low, close, volume, ...]
    2) Add advanced indicators
    3) Prepare data
    4) Train model
    5) Evaluate + Simple backtest
    """

    # -------------------------------------------------------------------------
    # 1) LOAD AND PREP DATA
    # -------------------------------------------------------------------------
    # csv_path = "data/your_crypto_data.csv"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, 'data', 'processed', 'BTC-USD_1d_processed.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Please update the path.")
        exit()

    df = pd.read_csv(csv_path)
    # rename columns to lowercase for convenience
    df.rename(columns=str.lower, inplace=True)
    # ensure ascending order by timestamp
    df.sort_values("timestamp", inplace=True)

    # Add advanced indicators
    df = add_advanced_indicators(df)

    # Set up which columns to treat as "price" vs. "indicator"
    price_cols = ["open", "high", "low", "close", "volume"]
    # example new indicator columns
    # indicator_cols = ["Stochrsi_k", "tochrsi_d", "adx", "obv", "atr_14"]
    indicator_cols = [
        'StochRSI_K', 'StochRSI_D', 'ADX', 'OBV', 'ATR_14'
    ]

    # Prepare data for a 5-day horizon label, 30-day LSTM window
    horizon = 5
    window_size = 30
    X_price, X_ind, X_seq, y, close_prices = prepare_data_for_fen(
        df, price_cols, indicator_cols, horizon=horizon, window_size=window_size
    )

    # Split train/test (time-based)
    split_idx = int(len(X_price) * 0.8)
    X_price_train, X_price_test = X_price[:split_idx], X_price[split_idx:]
    X_ind_train, X_ind_test = X_ind[:split_idx], X_ind[split_idx:]
    X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    close_train, close_test = close_prices[:split_idx], close_prices[split_idx:]

    print(f"Train samples: {len(X_price_train)}, Test samples: {len(X_price_test)}")

    # -------------------------------------------------------------------------
    # 2) BUILD AND TRAIN FEN MODEL
    # -------------------------------------------------------------------------
    model = build_fen_model(
        input_dim_price=X_price_train.shape[1],
        input_dim_indicators=X_ind_train.shape[1],
        seq_length=X_seq_train.shape[1],
        seq_features=X_seq_train.shape[2],
        hidden_units=64,
        l2_reg=0.001,
        dropout_rate=0.2
    )

    model.summary()

    history = model.fit(
        [X_price_train, X_ind_train, X_seq_train],
        y_train,
        validation_data=([X_price_test, X_ind_test, X_seq_test], y_test),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # Evaluate classification performance
    test_loss, test_acc = model.evaluate(
        [X_price_test, X_ind_test, X_seq_test],
        y_test,
        verbose=0
    )
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # -------------------------------------------------------------------------
    # 3) BACKTEST
    # -------------------------------------------------------------------------
    preds_test = model.predict([X_price_test, X_ind_test, X_seq_test]).flatten()

    # Example: daily re-balance, cost=0.0005
    daily_returns, cumrets, win_rate, final_cum_ret, sharpe_val = simple_backtest(
        predictions=preds_test,
        close_prices=close_test,
        horizon=1,       # trade daily (even though horizon=5 label)
        cost=0.0005
    )

    print(f"Strategy Win Rate: {win_rate*100:.2f}%")
    print(f"Strategy Annualized Sharpe: {sharpe_val:.2f}")
    print(f"Final Cumulative Return: {final_cum_ret*100:.2f}%")

    # Inspect sample predictions vs. actual labels
    print("\nSample predictions:", preds_test[:10])
    print("Sample test labels:", y_test[:10])
