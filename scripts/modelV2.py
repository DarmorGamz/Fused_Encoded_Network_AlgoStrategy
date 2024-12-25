import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

###############################################################################
# PART A. INDICATOR CALCULATIONS (Local Implementations)
###############################################################################

def compute_sma(series, window=50):
    """
    Simple Moving Average
    """
    return series.rolling(window=window, min_periods=window).mean()

def compute_ema(series, window=20):
    """
    Exponential Moving Average
    """
    return series.ewm(span=window, adjust=False).mean()

def compute_rsi(series, window=14):
    """
    Relative Strength Index (RSI) using the classic Wilder's formula.
    """
    # 1. Compute price changes
    delta = series.diff()

    # 2. Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # 3. Compute exponential moving averages of gains/losses
    avg_gain = gains.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False).mean()

    # 4. Calculate RS
    rs = avg_gain / avg_loss

    # 5. Calculate RSI
    rsi = 100 - (100 / (1.0 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    MACD: difference between fast EMA and slow EMA.
    Also compute the MACD Signal line (EMA of MACD).
    Returns: macd_line, macd_signal
    """
    ema_fast = compute_ema(series, window=fast)
    ema_slow = compute_ema(series, window=slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal

def compute_bollinger_bands(series, window=20, num_std=2):
    """
    Bollinger Bands (Upper and Lower) with a given window and STD multiplier.
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
    """
    Calculate a variety of technical indicators locally (no external library).
    Adjust or add more indicators as needed.
    """
    # We drop the first rows where indicators will be NaN
    # (especially for bigger windows like 50 for SMA).
    # We'll remove them at the end if needed.

    df['RSI'] = compute_rsi(df['close'], window=14)

    df['MACD'], df['MACD_Signal'] = compute_macd(df['close'], fast=12, slow=26, signal=9)

    df['BB_High'], df['BB_Low'] = compute_bollinger_bands(df['close'], window=20, num_std=2)

    df['EMA_20'] = compute_ema(df['close'], window=20)
    df['SMA_50'] = compute_sma(df['close'], window=50)

    # Drop any rows with NaNs that appear due to indicator calculations
    df.dropna(inplace=True)
    return df

def add_dummy_fundamental_features(df):
    """
    Dummy function to simulate fundamental or cross-sectional data.
    In real scenarios, you'd merge in actual fundamental metrics or external data.
    """
    np.random.seed(42)
    df['PE_Ratio'] = np.random.normal(15, 3, size=len(df))   # Fake P/E ratio
    df['Sentiment'] = np.random.uniform(-1, 1, size=len(df)) # Fake sentiment
    return df

def scale_features(df, price_cols, indicator_cols, extra_cols=None):
    """
    Scales selected columns using StandardScaler. Splits them out so each can be used in
    the correct model input branch.
    """
    if extra_cols is None:
        extra_cols = []

    all_cols = price_cols + indicator_cols + extra_cols
    scaler = StandardScaler()

    df[all_cols] = scaler.fit_transform(df[all_cols])
    return df, scaler

def prepare_fen_data(
    df,
    window_size=30,
    price_cols=('Open', 'High', 'Low', 'close', 'Volume'),
    indicator_cols=None,
    extra_cols=None
):
    """
    Prepares data for the Fused Encoded Network.
    Returns:
      X_price, X_indicators, X_seq, X_extra, y
    """
    if indicator_cols is None:
        # Example set of local-calculated indicator columns
        indicator_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'EMA_20', 'SMA_50'
        ]
    if extra_cols is None:
        extra_cols = ['PE_Ratio', 'Sentiment']

    # Create target: 1 if next close > current close, else 0
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop the last row if shift() caused an NaN in Target
    df.dropna(inplace=True)

    # Scale everything
    df, _ = scale_features(df, list(price_cols), list(indicator_cols), extra_cols)

    # Build arrays
    price_array = df[list(price_cols)].values
    indicator_array = df[list(indicator_cols)].values
    extra_array = df[list(extra_cols)].values  # e.g., fundamental / sentiment data

    # Create time-series sequences for LSTM input from the (scaled) price data
    seq_data = []
    for i in range(len(df) - window_size):
        seq_slice = price_array[i : i + window_size]
        seq_data.append(seq_slice)
    seq_data = np.array(seq_data)

    # Align the target with the sequences
    y = df['Target'].values[window_size:]

    # Align the other single-time inputs
    price_array = price_array[window_size:]
    indicator_array = indicator_array[window_size:]
    extra_array = extra_array[window_size:]

    return price_array, indicator_array, seq_data, extra_array, y

###############################################################################
# PART C. FUSED ENCODED NETWORK MODEL
###############################################################################

def build_fen_model(
    input_dim_price=5,         # e.g.  [Open, High, Low, close, Volume]
    input_dim_indicators=7,    # e.g.  [RSI, MACD, etc...]
    input_dim_extra=2,         # e.g.  [PE_Ratio, Sentiment]
    seq_length=30,             # window size
    seq_features=5,            # same as input_dim_price in this example
    task_type='binary',        # 'binary', 'multi', or 'regression'
    num_classes=1,             # used if task_type='multi'
    hidden_units=64,
    l2_reg=0.001,
    dropout_rate=0.2
):
    """
    Builds a Fused Encoded Network (FEN) with:
      - Price input (dense encoding)
      - Technical indicators input (dense encoding)
      - Sequential price data (LSTM + Multi-Head Self-Attention)
      - Extra/fundamental/sentiment input (dense encoding)
      - Attention-based fusion
      - Output: binary/multi-class/regression
    """

    ###########################
    # 1. Price Input Branch
    ###########################
    price_input = layers.Input(shape=(input_dim_price,), name='Price_Input')
    x_price = layers.Dense(hidden_units, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(price_input)
    x_price = layers.BatchNormalization()(x_price)
    x_price = layers.Dropout(dropout_rate)(x_price)
    x_price = layers.Dense(hidden_units//2, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(x_price)
    x_price = layers.BatchNormalization()(x_price)

    ###########################
    # 2. Technical Indicator Branch
    ###########################
    indicator_input = layers.Input(shape=(input_dim_indicators,), name='Indicator_Input')
    x_ind = layers.Dense(hidden_units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(indicator_input)
    x_ind = layers.BatchNormalization()(x_ind)
    x_ind = layers.Dropout(dropout_rate)(x_ind)
    x_ind = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_ind)
    x_ind = layers.BatchNormalization()(x_ind)

    ###########################
    # 3. Sequential (Time-Series) Branch
    ###########################
    seq_input = layers.Input(shape=(seq_length, seq_features), name='Sequential_Input')
    x_seq = layers.LSTM(hidden_units, return_sequences=True,
                        kernel_regularizer=regularizers.l2(l2_reg))(seq_input)
    x_seq = layers.BatchNormalization()(x_seq)
    
    # Multi-Head Self-Attention: self-attention over time dimension
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=hidden_units//4)(
        x_seq, x_seq
    )
    # Residual connection + layer normalization
    x_seq = layers.Add()([x_seq, attn])
    x_seq = layers.LayerNormalization()(x_seq)

    # Pool across time
    x_seq = layers.GlobalAveragePooling1D()(x_seq)
    x_seq = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x_seq)
    x_seq = layers.BatchNormalization()(x_seq)

    ###########################
    # 4. Extra / Fundamental / Cross-Sectional Branch
    ###########################
    extra_input = layers.Input(shape=(input_dim_extra,), name='Extra_Input')
    x_extra = layers.Dense(hidden_units//2, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(extra_input)
    x_extra = layers.BatchNormalization()(x_extra)
    x_extra = layers.Dropout(dropout_rate)(x_extra)
    x_extra = layers.Dense(hidden_units//4, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg))(x_extra)
    x_extra = layers.BatchNormalization()(x_extra)

    ###########################
    # 5. Fusion
    ###########################
    fused = layers.Concatenate(axis=-1)([x_price, x_ind, x_seq, x_extra])
    fused = layers.Dense(hidden_units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(dropout_rate)(fused)
    fused = layers.Dense(hidden_units//2, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(fused)
    fused = layers.BatchNormalization()(fused)

    ###########################
    # 6. Output
    ###########################
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

    # Build and compile
    model = models.Model(
        inputs=[price_input, indicator_input, seq_input, extra_input],
        outputs=output
    )
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss, metrics=metrics)

    return model

###############################################################################
# PART D. EXAMPLE USAGE
###############################################################################

if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # 1) LOAD DATA (replace this with your actual data)
    # ---------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(BASE_DIR, 'data', 'processed', 'BTC-USD_1d_processed.csv')

    # data_file = 'data/your_equity_or_crypto.csv'
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}. Please specify a valid CSV path.")
        exit()

    df = pd.read_csv(data_file, parse_dates=True, index_col=0)
    df.sort_index(inplace=True)  # Ensure ascending date order

    # ---------------------------------------------------------------------
    # 2) ADD TECHNICAL + FUNDAMENTAL INDICATORS
    # ---------------------------------------------------------------------
    df = add_technical_indicators(df)
    df = add_dummy_fundamental_features(df)

    # ---------------------------------------------------------------------
    # 3) PREPARE DATA FOR FEN
    # ---------------------------------------------------------------------
    window_size = 30
    price_cols = ('open', 'high', 'low', 'close', 'volume')
    indicator_cols = [
        'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'EMA_20', 'SMA_50'
    ]
    extra_cols = ['PE_Ratio', 'Sentiment']

    X_price, X_ind, X_seq, X_extra, y = prepare_fen_data(
        df,
        window_size=window_size,
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

    print(f"Train samples: {len(X_price_train)}, Test samples: {len(X_price_test)}")

    # ---------------------------------------------------------------------
    # 5) BUILD AND TRAIN THE FEN MODEL
    # ---------------------------------------------------------------------
    model = build_fen_model(
        input_dim_price=X_price_train.shape[1],
        input_dim_indicators=X_ind_train.shape[1],
        input_dim_extra=X_extra_train.shape[1],
        seq_length=X_seq_train.shape[1],
        seq_features=X_seq_train.shape[2],
        task_type='binary',      # or 'multi'/'regression'
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
    # 6) EVALUATE AND PREDICT
    # ---------------------------------------------------------------------
    loss, metric_val = model.evaluate(
        [X_price_test, X_ind_test, X_seq_test, X_extra_test],
        y_test,
        verbose=0
    )

    if model.output_shape[-1] == 1 and model.layers[-1].activation.__name__ == 'sigmoid':
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {metric_val:.4f}")
    else:
        print(f"Test Loss: {loss:.4f}, Test Metric ({model.metrics_names[1]}): {metric_val:.4f}")

    preds = model.predict([X_price_test[:5], X_ind_test[:5], X_seq_test[:5], X_extra_test[:5]]).flatten()
    print("Sample predictions:", preds)
    print("Sample targets:", y_test[:5])
