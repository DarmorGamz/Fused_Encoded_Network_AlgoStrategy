# scripts/backtest.py

import numpy as np
import pandas as pd

def backtest_strategy(df, preds, window_size=30, initial_capital=10000):
    """
    Simple backtest of a long/cash strategy.
    
    Args:
        df: DataFrame with the original OHLC data
        preds: Model predictions aligned with the DF after window adjustment
        window_size: The lookback window used in training
        initial_capital: starting capital
    
    Returns:
        A DataFrame with equity curve
    """
    df_backtest = df.iloc[window_size:].copy()
    df_backtest['Signal'] = (preds > 0.5).astype(int)
    
    # Let's assume we go fully long if signal=1, or go fully in cash if signal=0
    df_backtest['Pct_Change'] = df_backtest['Close'].pct_change()
    df_backtest['Strategy_Return'] = df_backtest['Signal'] * df_backtest['Pct_Change']
    df_backtest['Equity'] = (1 + df_backtest['Strategy_Return']).cumprod() * initial_capital
    
    return df_backtest

if __name__ == "__main__":
    # Example usage
    from model import prepare_training_data, build_fused_model
    
    processed_file = 'data/processed/BTC-USD_1d_processed.csv'
    df = pd.read_csv(processed_file, parse_dates=True, index_col=0)
    
    X_price, X_indicators, X_sequence, y = prepare_training_data(df, window_size=30)
    
    split_idx = int(len(X_price)*0.8)
    X_price_train, X_price_test = X_price[:split_idx], X_price[split_idx:]
    X_ind_train, X_ind_test = X_indicators[:split_idx], X_indicators[split_idx:]
    X_seq_train, X_seq_test = X_sequence[:split_idx], X_sequence[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = build_fused_model(
        input_dim_price=X_price_train.shape[1],
        input_dim_indicators=X_ind_train.shape[1]
    )
    
    # Normally you'd load a saved model or just do a quick training here
    model.fit([X_price_train, X_ind_train, X_seq_train], y_train, epochs=1, batch_size=32)
    
    preds = model.predict([X_price_test, X_ind_test, X_seq_test])
    df_backtest = backtest_strategy(df, preds, window_size=30)
    
    # Evaluate results
    final_equity = df_backtest['Equity'].iloc[-1]
    print(f"Final Equity: {final_equity}")
    df_backtest[['Close', 'Signal', 'Equity']].tail(10).to_csv("backtest_results.csv")
