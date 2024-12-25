# scripts/model.py

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import layers, models

from sklearn.preprocessing import StandardScaler


def build_fused_model(
    input_dim_price=5,         # e.g., [Open, High, Low, Close, Volume]
    input_dim_indicators=6,    # e.g., [SMA_14, EMA_14, RSI_14, MACD, etc.]
    input_dim_cross=1,         # e.g., [Momentum_Rank] or multiple cross-sectional features
    sequence_length=30,        # e.g., 30 time steps for LSTM
    sequence_features=5,       # e.g., same 5 as in input_dim_price, or more
    num_classes=1,             # 1 for binary classification, or more for multi-class
    task_type='binary'         # 'binary', 'multi', or 'regression'
):
    """
    Builds a more advanced Fused Encoder Network model that:
      1) Integrates price data
      2) Integrates technical indicators
      3) Integrates time-series data with an attention-based LSTM
      4) Integrates cross-sectional features (e.g. momentum rank)
      5) Uses advanced regularization (BatchNorm, L2)
      6) Can output either binary, multi-class, or regression predictions
    """

    # ==================== Branch A: Price Data ====================
    price_input = layers.Input(shape=(input_dim_price,), name="Price_Input")
    x1 = layers.Dense(
        64, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(price_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.Dense(
        32, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x1)
    x1 = layers.BatchNormalization()(x1)

    # ==================== Branch B: Technical Indicators ====================
    indicator_input = layers.Input(shape=(input_dim_indicators,), name="Indicator_Input")
    x2 = layers.Dense(
        64, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(indicator_input)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(
        32, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x2)
    x2 = layers.BatchNormalization()(x2)

    # ==================== Branch C: Sequential Data (Attention) ====================
    seq_input = layers.Input(shape=(sequence_length, sequence_features), name="Sequential_Input")
    x3 = layers.LSTM(
        64, return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(seq_input)

    # Self-Attention: x3 attends to itself
    attention = layers.Attention()([x3, x3])
    # Average pooling across time dimension
    x3 = layers.GlobalAveragePooling1D()(attention)

    x3 = layers.Dense(
        32, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x3)
    x3 = layers.BatchNormalization()(x3)

    # ==================== Branch D: Cross-Sectional Features ====================
    cross_section_input = layers.Input(shape=(input_dim_cross,), name="Cross_Section_Input")
    x4 = layers.Dense(
        32, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(cross_section_input)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.2)(x4)
    x4 = layers.Dense(
        16, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x4)
    x4 = layers.BatchNormalization()(x4)

    # ==================== Fuse All Branches ====================
    fused = layers.concatenate([x1, x2, x3, x4], axis=-1)
    fused = layers.Dense(
        64, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.2)(fused)
    fused = layers.Dense(
        32, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(fused)
    fused = layers.BatchNormalization()(fused)

    # ==================== Output Layer ====================
    if task_type == 'binary':
        # Binary classification
        output = layers.Dense(1, activation='sigmoid', name="Output")(fused)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task_type == 'multi':
        # Multi-class classification
        output = layers.Dense(num_classes, activation='softmax', name="Output")(fused)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        # Regression
        output = layers.Dense(1, activation='linear', name="Output")(fused)
        loss = 'mse'
        metrics = ['mae']

    model = models.Model(
        inputs=[price_input, indicator_input, seq_input, cross_section_input],
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=metrics
    )

    return model


def scale_features(df, price_cols, indicator_cols):
    """
    Scales the specified price and indicator columns using StandardScaler.
    Returns:
        The dataframe with scaled columns in place.
        (In practice, you might separate scaler fit on train and transform on test.)
    """
    scaler = StandardScaler()
    # Fit_transform all in one for simplicity;
    # ideally, fit on train set only, then transform test set.
    df[price_cols + indicator_cols] = scaler.fit_transform(df[price_cols + indicator_cols])
    return df


def prepare_training_data(df, window_size=30):
    """
    Prepares data for model training.
    Returns:
        X_price, X_indicators, X_sequence, y
        (You will also need to prepare X_cross separately if using cross-sectional features.)
    """
    # Create binary target: next close price > current close => 1, else 0
    df["Target"] = ((df["Close"].shift(-1) > df["Close"]) * 1).fillna(0)

    # Drop any NaNs (e.g., from shift)
    df.dropna(inplace=True)

    # Quick label check
    print("Unique target labels:", df["Target"].unique())

    # Define columns
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # If you have these indicator columns, adjust to match your dataset
    indicator_cols = [
        col for col in df.columns
        if col not in price_cols + ['Target']
    ]

    # Scale features (simple approach: scale entire df at once)
    df = scale_features(df, price_cols, indicator_cols)

    # Now extract numpy arrays for model inputs
    X_price = df[price_cols].values
    X_indicators = df[indicator_cols].values

    # Build the sequence data
    X_sequence = []
    for i in range(len(df) - window_size):
        seq_slice = df.iloc[i : i+window_size][price_cols].values
        X_sequence.append(seq_slice)
    X_sequence = np.array(X_sequence)

    # Align y
    y = df['Target'].values[window_size:]

    # Also align the other inputs
    X_price = X_price[window_size:]
    X_indicators = X_indicators[window_size:]

    return X_price, X_indicators, X_sequence, y


if __name__ == "__main__":
    # Example usage
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_file = os.path.join(BASE_DIR, 'data', 'processed', 'BTC-USD_1d_processed.csv')

    df = pd.read_csv(processed_file, parse_dates=True, index_col=0)

    # Prepare main data
    X_price, X_indicators, X_sequence, y = prepare_training_data(df, window_size=30)

    # -------------------------------------------------------------------
    #  EXAMPLE: CROSS-SECTIONAL FEATURES (X_cross)
    #  If you don't have cross-sectional features, either skip or create a dummy array:
    #  For instance, a single feature (momentum rank):
    #     X_cross = df["Momentum_Rank"].values
    #  Make sure to align it as well if it depends on the same window_size offset.
    # -------------------------------------------------------------------
    # In this example, let's just create a dummy cross array of zeros.
    # Normally, you'd have something like df["Momentum_Rank"] or other cross-asset data.
    dummy_cross_col = np.zeros_like(df["Close"].values)
    X_cross = dummy_cross_col[30:]  # align with the same window_size offset

    # Now we have X_cross as shape (num_samples,) but
    # the model expects 2D (batch_size, input_dim_cross).
    # Let's reshape if needed:
    X_cross = X_cross.reshape(-1, 1)  # input_dim_cross=1

    # Train/test split (time-based)
    split_idx = int(len(X_price) * 0.8)
    X_price_train, X_price_test = X_price[:split_idx], X_price[split_idx:]
    X_ind_train, X_ind_test = X_indicators[:split_idx], X_indicators[split_idx:]
    X_seq_train, X_seq_test = X_sequence[:split_idx], X_sequence[split_idx:]
    X_cross_train, X_cross_test = X_cross[:split_idx], X_cross[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train size: {len(X_price_train)}, Test size: {len(X_price_test)}")

    # Build the fused model with Cross-Section Input
    model = build_fused_model(
        input_dim_price=X_price_train.shape[1],
        input_dim_indicators=X_ind_train.shape[1],
        input_dim_cross=X_cross_train.shape[1],  # cross features dimension
        sequence_length=X_seq_train.shape[1],
        sequence_features=X_seq_train.shape[2],
        task_type='binary'
    )

    # Example training - pass four inputs: price, indicators, seq, cross
    model.fit(
        [X_price_train, X_ind_train, X_seq_train, X_cross_train],
        y_train,
        validation_data=([X_price_test, X_ind_test, X_seq_test, X_cross_test], y_test),
        epochs=20,
        batch_size=32
    )

    # Evaluate - you must also pass four inputs for evaluation
    test_loss, test_acc = model.evaluate([X_price_test, X_ind_test, X_seq_test, X_cross_test], y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    # Optional: Inspect predictions
    preds = model.predict([X_price_test[:5], X_ind_test[:5], X_seq_test[:5], X_cross_test[:5]]).flatten()
    print("Sample predictions:", preds)
    print("Sample targets:", y_test[:5])
