import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

###############################################################################
# PART A. INDICATOR CALCULATIONS
###############################################################################

def compute_sma(series, window=50):
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()

def compute_ema(series, window=20):
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()

def compute_rsi(series, window=14):
    """
    Relative Strength Index (RSI) using the classic Wilder's formula.
    Returns RSI in [0, 100].
    """
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1.0 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    MACD: difference between fast EMA and slow EMA.
    Returns (macd_line, macd_signal).
    """
    ema_fast = compute_ema(series, window=fast)
    ema_slow = compute_ema(series, window=slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal

def compute_bollinger_bands(series, window=20, num_std=2):
    """
    Bollinger Bands (Upper and Lower).
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    bb_high = rolling_mean + (rolling_std * num_std)
    bb_low = rolling_mean - (rolling_std * num_std)
    return bb_high, bb_low

###############################################################################
# PART B. DATA PREPARATION
###############################################################################

def add_technical_indicators(df):
    """Local calculation of common indicators."""
    df['RSI'] = compute_rsi(df['close'], window=14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['close'], fast=12, slow=26, signal=9)
    df['BB_High'], df['BB_Low'] = compute_bollinger_bands(df['close'], window=20, num_std=2)
    df['EMA_20'] = compute_ema(df['close'], window=20)
    df['SMA_50'] = compute_sma(df['close'], window=50)
    df.dropna(inplace=True)
    return df

def add_dummy_fundamental_features(df):
    """
    Dummy function simulating fundamental or sentiment data. 
    Replace with real fundamentals, cross-sectional signals, or textual sentiment.
    """
    np.random.seed(42)
    df['PE_Ratio'] = np.random.normal(15, 3, size=len(df))   # Fake P/E
    df['Sentiment'] = np.random.uniform(-1, 1, size=len(df)) # Fake sentiment
    return df

def scale_features(df, price_cols, indicator_cols, extra_cols=None):
    """
    Scales selected columns using StandardScaler.
    """
    if extra_cols is None:
        extra_cols = []
    all_cols = price_cols + indicator_cols + extra_cols
    scaler = StandardScaler()
    df[all_cols] = scaler.fit_transform(df[all_cols])
    return df, scaler

def create_multiday_label(df, horizon=5):
    """
    Label = 1 if close[t+horizon] > close[t], else 0.
    This is a multi-day horizon label, often used in FEN research.
    """
    df['Target'] = (df['close'].shift(-horizon) > df['close']).astype(int)

def prepare_fen_data(
    df,
    window_size=30,
    horizon=5,
    price_cols=('Open', 'High', 'Low', 'close', 'Volume'),
    indicator_cols=None,
    extra_cols=None
):
    """
    Prepares data for the Fused Encoded Network with a multi-day horizon target.
    """
    if indicator_cols is None:
        indicator_cols = ['RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'EMA_20', 'SMA_50']
    if extra_cols is None:
        extra_cols = ['PE_Ratio', 'Sentiment']

    # Create the multi-day horizon label
    create_multiday_label(df, horizon=horizon)
    df.dropna(inplace=True)  # remove rows with NaN from shift or indicators

    # Scale relevant columns
    df, _ = scale_features(df, list(price_cols), list(indicator_cols), extra_cols)

    # Build arrays
    price_array = df[list(price_cols)].values
    indicator_array = df[list(indicator_cols)].values
    extra_array = df[list(extra_cols)].values  # e.g. fundamental / sentiment
    
    # Build sequence data from the scaled price array
    seq_data = []
    for i in range(len(df) - window_size):
        seq_slice = price_array[i : i + window_size]
        seq_data.append(seq_slice)
    seq_data = np.array(seq_data)

    # Align labels with sequences
    y = df['Target'].values[window_size:]

    # Also align the single-time inputs
    price_array = price_array[window_size:]
    indicator_array = indicator_array[window_size:]
    extra_array = extra_array[window_size:]

    # Track the close prices for backtesting later
    # (We shift them as well to match the labels)
    close_prices = df['close'].values[window_size:]

    return price_array, indicator_array, seq_data, extra_array, y, close_prices

###############################################################################
# PART C. FUSED ENCODED NETWORK MODEL
###############################################################################

def build_fen_model(
    input_dim_price=5,
    input_dim_indicators=7,
    input_dim_extra=2,
    seq_length=30,
    seq_features=5,
    task_type='binary',
    num_classes=1,
    hidden_units=64,
    l2_reg=0.001,
    dropout_rate=0.2
):
    """
    Fused Encoded Network with:
      - Price input
      - Technical indicators input
      - LSTM + Multi-Head Attention for sequential data
      - Extra/fundamental/sentiment
      - Output: binary, multi, or regression
    """
    # 1. Price Branch
    price_input = layers.Input(shape=(input_dim_price,), name='Price_Input')
    x_price = layers.Dense(hidden_units, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(price_input)
    x_price = layers.BatchNormalization()(x_price)
    x_price = layers.Dropout(dropout_rate)(x_price)
    x_price = layers.Dense(hidden_units//2, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(x_price)
    x_price = layers.BatchNormalization()(x_price)

    # 2. Indicator Branch
    indicator_input = layers.Input(shape=(input_dim_indicators,), name='Indicator_Input')
    x_ind = layers.Dense(hidden_units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(indicator_input)
    x_ind = layers.BatchNormalization()(x_ind)
    x_ind = layers.Dropout(dropout_rate)(x_ind)
    x_ind = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_ind)
    x_ind = layers.BatchNormalization()(x_ind)

    # 3. Sequential Branch
    seq_input = layers.Input(shape=(seq_length, seq_features), name='Sequential_Input')
    x_seq = layers.LSTM(hidden_units, return_sequences=True,
                        kernel_regularizer=regularizers.l2(l2_reg))(seq_input)
    x_seq = layers.BatchNormalization()(x_seq)

    # Multi-Head Self-Attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=hidden_units//4)(x_seq, x_seq)
    x_seq = layers.Add()([x_seq, attn])  # Residual
    x_seq = layers.LayerNormalization()(x_seq)
    x_seq = layers.GlobalAveragePooling1D()(x_seq)
    x_seq = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_seq)
    x_seq = layers.BatchNormalization()(x_seq)

    # 4. Extra / Fundamental Branch
    extra_input = layers.Input(shape=(input_dim_extra,), name='Extra_Input')
    x_extra = layers.Dense(hidden_units//2, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(extra_input)
    x_extra = layers.BatchNormalization()(x_extra)
    x_extra = layers.Dropout(dropout_rate)(x_extra)
    x_extra = layers.Dense(hidden_units//4, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(x_extra)
    x_extra = layers.BatchNormalization()(x_extra)

    # 5. Fusion
    fused = layers.Concatenate(axis=-1)([x_price, x_ind, x_seq, x_extra])
    fused = layers.Dense(hidden_units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(dropout_rate)(fused)
    fused = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)

    # 6. Output Layer
    if task_type == 'binary':
        output = layers.Dense(1, activation='sigmoid', name='Output')(fused)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task_type == 'multi':
        output = layers.Dense(num_classes, activation='softmax', name='Output')(fused)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:  # regression
        output = layers.Dense(1, activation='linear', name='Output')(fused)
        loss = 'mse'
        metrics = ['mae']

    model = models.Model(
        inputs=[price_input, indicator_input, seq_input, extra_input],
        outputs=output
    )
    model.compile(optimizer=Adam(1e-4), loss=loss, metrics=metrics)
    return model

###############################################################################
# PART D. SIMPLE BACKTESTING AND SHARPE METRICS
###############################################################################

def simple_backtest(predictions, close_prices, horizon=1, cost=0.0005):
    """
    A simple daily backtest:
      - predictions > 0.5 => go long for the next day; predictions < 0.5 => go short
      - horizon: how many days you hold the position (1 by default)
      - cost: transaction cost (round-turn). E.g. 0.0005 => 0.05% each trade.

    Returns:
      - daily_returns (pd.Series): daily returns from the strategy
      - cumulative_returns (pd.Series): cumulative product of (1 + daily_returns)
      - win_rate (float)
    """
    # Ensure shapes match
    n_preds = len(predictions)
    n_prices = len(close_prices)
    if n_preds != n_prices:
        raise ValueError(f"Prediction length ({n_preds}) != close_prices length ({n_prices})")

    # We shift close_prices by 1 so that each prediction is used to trade the *next* day
    # Alternatively, you might do no shift if you’re using “open” next day or intraday data.
    # Adjust logic depending on how your data is structured.
    shifted_close = pd.Series(close_prices).shift(-1 * horizon)
    
    # Binary trading signal: +1 if > 0.5, -1 otherwise
    signal = np.where(predictions > 0.5, 1.0, -1.0)
    
    # Percentage change for each day (or horizon).
    # close[t+horizon]/close[t] - 1
    # We'll do a simple approach: daily return is (close[t+1] - close[t]) / close[t].
    # For horizon>1, we do close[t+horizon] / close[t] - 1.
    current_close = pd.Series(close_prices)
    ret = (shifted_close.values - current_close.values) / current_close.values
    
    # Strategy return = signal * market return
    strat_ret = signal * ret
    
    # Subtract transaction cost each time we flip from +1 to -1 or -1 to +1.
    # For simplicity, assume we open/close a position each day => cost is incurred daily.
    # If horizon=5, you might adapt the cost logic so cost is only once every 5 days, etc.
    strat_ret -= cost

    # Convert to pandas
    strat_ret = pd.Series(strat_ret, index=current_close.index)
    
    # Drop last horizon rows that have NaN because of shifting
    strat_ret.dropna(inplace=True)
    
    # Calculate daily metrics
    daily_returns = strat_ret
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # Win rate
    wins = daily_returns[daily_returns > 0].count()
    total_trades = daily_returns.count()
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return daily_returns, cumulative_returns, win_rate

def annualized_sharpe(daily_returns, trading_days=252):
    """
    Annualized Sharpe Ratio = (mean(daily_returns)*trading_days) / (std(daily_returns)*sqrt(trading_days))
    """
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    if std_ret == 0:
        return 0.0
    return (mean_ret * trading_days) / (std_ret * np.sqrt(trading_days))

###############################################################################
# PART E. EXAMPLE USAGE
###############################################################################

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1) LOAD DATA (replace with your data)
    # ---------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(BASE_DIR, 'data', 'processed', 'BTC-USD_1d_processed.csv')
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}. Update data_file path.")
        exit()

    df = pd.read_csv(data_file, parse_dates=True, index_col=0)
    df.sort_index(inplace=True)  # ascending date order

    # ---------------------------------------------------------------------
    # 2) ADD TECHNICAL + FUNDAMENTAL INDICATORS
    # ---------------------------------------------------------------------
    df = add_technical_indicators(df)
    df = add_dummy_fundamental_features(df)

    # ---------------------------------------------------------------------
    # 3) PREPARE FEN DATA
    # ---------------------------------------------------------------------
    # Example: 5-day horizon target => "Will close[t+5] > close[t]?"
    window_size = 30
    horizon = 5
    price_cols = ('open', 'high', 'low', 'close', 'volume')
    indicator_cols = ['RSI','MACD','MACD_Signal','BB_High','BB_Low','EMA_20','SMA_50']
    extra_cols = ['PE_Ratio','Sentiment']

    X_price, X_ind, X_seq, X_extra, y, close_prices = prepare_fen_data(
        df,
        window_size=window_size,
        horizon=horizon,
        price_cols=price_cols,
        indicator_cols=indicator_cols,
        extra_cols=extra_cols
    )

    # ---------------------------------------------------------------------
    # 4) TRAIN/TEST SPLIT
    # ---------------------------------------------------------------------
    split_idx = int(len(X_price) * 0.8)
    X_price_train, X_price_test = X_price[:split_idx], X_price[split_idx:]
    X_ind_train, X_ind_test = X_ind[:split_idx], X_ind[split_idx:]
    X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
    X_extra_train, X_extra_test = X_extra[:split_idx], X_extra[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # For backtest, we also split close prices the same way
    close_train, close_test = close_prices[:split_idx], close_prices[split_idx:]
    
    print(f"Train samples: {len(X_price_train)}, Test samples: {len(X_price_test)}")

    # ---------------------------------------------------------------------
    # 5) BUILD AND TRAIN MODEL
    # ---------------------------------------------------------------------
    model = build_fen_model(
        input_dim_price=X_price_train.shape[1],
        input_dim_indicators=X_ind_train.shape[1],
        input_dim_extra=X_extra_train.shape[1],
        seq_length=X_seq_train.shape[1],
        seq_features=X_seq_train.shape[2],
        task_type='binary',  
        hidden_units=64,
        l2_reg=0.001,
        dropout_rate=0.2
    )
    model.summary()

    history = model.fit(
        [X_price_train, X_ind_train, X_seq_train, X_extra_train],
        y_train,
        validation_data=([X_price_test, X_ind_test, X_seq_test, X_extra_test], y_test),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # ---------------------------------------------------------------------
    # 6) EVALUATE CLASSIFICATION ACCURACY
    # ---------------------------------------------------------------------
    loss, accuracy = model.evaluate(
        [X_price_test, X_ind_test, X_seq_test, X_extra_test], 
        y_test,
        verbose=0
    )
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # ---------------------------------------------------------------------
    # 7) BACKTEST: PREDICT, COMPUTE RETURNS, SHARPE, WIN RATE
    # ---------------------------------------------------------------------
    preds_test = model.predict([X_price_test, X_ind_test, X_seq_test, X_extra_test]).flatten()
    
    # Simple backtest with daily horizon=1 trades
    # If your label horizon is 5, you might also hold 5 days. 
    # For illustration, let's do daily trades with cost=0.0005 (0.05%)
    daily_returns, cumret, win_rate = simple_backtest(
        predictions=preds_test,
        close_prices=close_test,
        horizon=1,       # trade daily (even though label used horizon=5)
        cost=0.0005
    )
    # Compute Sharpe
    sharpe_val = annualized_sharpe(daily_returns, trading_days=252)

    print(f"Strategy Win Rate: {win_rate*100:.2f}%")
    print(f"Strategy Annualized Sharpe: {sharpe_val:.2f}")

    # Optionally look at final cumulative return
    final_cumret = cumret.iloc[-1]
    print(f"Final Cumulative Return: {final_cumret - 1:.2%}")

    # Inspect sample predictions vs. actual labels
    print("\nSample predictions:", preds_test[:10])
    print("Sample test labels:", y_test[:10])
