import os
import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from binance.client import Client
from binance.enums import HistoricalKlinesType

###############################################################################
# 1. CONFIGURATION & DATA ACQUISITION
###############################################################################
API_KEY = "<YOUR_BINANCE_API_KEY>"
API_SECRET = "<YOUR_BINANCE_API_SECRET>"
client = Client(API_KEY, API_SECRET)

# According to the paper, the authors used 1-hour data over 3+ years.
# We’ll fetch BTCUSDT 1-hour candles as an example.
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
START_DATE = "1 Jan, 2021"
FILENAME = "BTCUSDT_AMML_1H.csv"

def fetch_data(symbol: str = SYMBOL, 
               interval: str = INTERVAL, 
               start: str = START_DATE,
               filename: str = FILENAME) -> pd.DataFrame:
    """
    1) Checks if local CSV exists, load if so.
    2) Otherwise, fetch from Binance and save CSV.
    Returns a DataFrame sorted by time with columns:
      ['open_time','open','high','low','close','volume',...]
    """
    if os.path.exists(filename):
        print(f"[INFO] Loading {symbol} data from {filename}")
        df = pd.read_csv(filename, parse_dates=['open_time'])
        return df
    else:
        print(f"[INFO] Fetching historical data for {symbol} from Binance...")
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start,
            klines_type=HistoricalKlinesType.SPOT
        )
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume','ignore'
        ])
        numeric_cols = ['open','high','low','close','volume',
                        'quote_asset_volume','taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.sort_values('open_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(filename, index=False)
        return df

# Load Data
raw_df = fetch_data()

###############################################################################
# 2. FEATURE ENGINEERING
###############################################################################
def build_features(df: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """
    Creates features based on the paper's "Adaptive Momentum + Order Imbalance" approach:
      - Rolling returns
      - Rolling volatility
      - Rolling volume spikes
      - (Optional) Best bid-ask imbalance (requires order-book data, 
        but here we'll approximate with volume-based features for brevity)
    The 'target' will be a binary variable indicating if the next bar's close is higher/lower.
    """
    # We'll keep a copy
    df_feat = df.copy()

    # Basic columns
    df_feat['returns'] = df_feat['close'].pct_change()
    df_feat['rolling_ret'] = df_feat['close'].pct_change(lookback)
    df_feat['rolling_vol'] = df_feat['returns'].rolling(lookback).std()
    df_feat['rolling_vol'].fillna(method='bfill', inplace=True)
    df_feat['rolling_volume'] = df_feat['volume'].rolling(lookback).mean()
    df_feat['rolling_volume'].fillna(method='bfill', inplace=True)

    # For classification: if next close > current close => 1, else 0
    # Shift -1 to represent "future" direction
    df_feat['future_close'] = df_feat['close'].shift(-1)
    df_feat['target'] = (df_feat['future_close'] > df_feat['close']).astype(int)
    df_feat.dropna(inplace=True)  # Drop the last row which has no future_close

    # The features we'll use:
    feature_cols = ['returns', 'rolling_ret', 'rolling_vol', 'rolling_volume']
    df_feat = df_feat[['open_time','close','target'] + feature_cols].copy()

    # Re-index by open_time for clarity
    df_feat.set_index('open_time', inplace=True)
    return df_feat

feat_df = build_features(raw_df, lookback=24)

###############################################################################
# 3. MODEL TRAINING & PREDICTION (WALK-FORWARD)
###############################################################################
def walk_forward_prediction(df: pd.DataFrame, 
                            feature_cols: list, 
                            retrain_every: int = 24 * 7, 
                            test_size: int = 24 * 7):
    """
    Implements the 'monthly' or 'weekly' rolling approach from the paper.
    The paper states: "We retrain the model every N bars using a rolling window."
    We'll do the following:
      1) Start with an initial training set (e.g., first X bars).
      2) Retrain every 'retrain_every' bars.
      3) Predict each bar in the test set, then slide forward.
    Returns a DataFrame of the same length with 'prediction_prob' (prob of upward move).
    """
    df_out = df.copy()
    df_out['prediction_prob'] = np.nan

    # The paper indicates ~2 years for initial training, then repeated weekly retraining.
    # We'll pick an initial training of 2000 bars (~3 months of 1-hour data).
    initial_train_len = 2000
    if len(df_out) < initial_train_len:
        raise ValueError("Not enough data for initial training window.")

    # Convert to numeric arrays
    X_all = df_out[feature_cols].values
    y_all = df_out['target'].values

    # We'll keep track of start/end
    start_idx = initial_train_len

    # The paper uses a gradient-boosted model (XGBoost/LightGBM). We'll choose XGBoost here.
    # We iterate from 'start_idx' to the end in increments of 1 (predict each bar).
    while start_idx < len(df_out) - test_size:
        # Train set: from 0 to start_idx
        X_train = X_all[:start_idx]
        y_train = y_all[:start_idx]

        # Next block to predict: from start_idx to start_idx + test_size
        end_idx = min(start_idx + test_size, len(df_out))
        X_test = X_all[start_idx:end_idx]

        # Train the model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Predict probabilities
        test_probs = model.predict_proba(X_test)[:,1]  # Probability of target=1
        df_out.iloc[start_idx:end_idx, df_out.columns.get_loc('prediction_prob')] = test_probs

        # Slide forward
        start_idx += test_size  # retrain again after the test_size window

    return df_out

feature_list = ['returns', 'rolling_ret', 'rolling_vol', 'rolling_volume']
pred_df = walk_forward_prediction(feat_df, feature_cols=feature_list)
print(pred_df.isnull().sum())

###############################################################################
# 4. BACKTEST (IDENTICAL TO THE RESEARCH PAPER)
###############################################################################
def backtest_amml(df: pd.DataFrame,
                  threshold_long: float = 0.6,
                  threshold_short: float = 0.4,
                  initial_capital: float = 10_000,
                  fee_rate: float = 0):
    """
    According to the paper’s procedure:
      - If model's predicted probability > threshold_long => go long 1 unit
      - If model's predicted probability < threshold_short => go short 1 unit
      - Otherwise => flat
    The paper uses "1 unit of base currency" (e.g., 1 BTC) for simplicity or 
    a fraction of capital. For demonstration, let's assume 1 BTC each trade.
    
    *Trade Execution*: The paper states "enter at next bar open" to avoid look-ahead.
    *Exit Condition*: The position is reversed or closed out when the next opposing signal arrives.
    """
    # Prepare for iteration
    df_bt = df.copy()
    df_bt['position'] = 0
    df_bt['pnl'] = 0.0
    df_bt['equity'] = 0.0
    
    pos = 0  # 1 => long, -1 => short, 0 => flat
    entry_price = 0.0
    cum_pnl = 0.0
    
    close_prices = df_bt['close'].values
    probs = df_bt['prediction_prob'].values

    for i in range(1, len(df_bt)):
        prev_prob = probs[i-1]  # we act on the previous bar's signal
        current_close = close_prices[i]

        # Determine signal based on threshold
        if prev_prob > threshold_long:
            signal = 1
        elif prev_prob < threshold_short:
            signal = -1
        else:
            signal = 0

        # Check if we need to exit or switch position
        if pos == 0:
            # If we're flat, open a position
            if signal != 0:
                pos = signal
                entry_price = current_close
        else:
            # If we already have a position
            if signal == 0 or signal == -pos:
                # Exit or reverse
                exit_price = current_close
                trade_pnl = (exit_price - entry_price) * pos
                # Subtract fees => we pay fees on the notional value of the trade
                # Notional = 1 coin * exit_price (approx). We apply fee on entry & exit => 2 trades
                fee = fee_rate * abs(exit_price) * 1.0 * 2  
                net_trade_pnl = trade_pnl - fee
                cum_pnl += net_trade_pnl
                
                # If reversing, open the new position immediately at same price
                if signal == -pos:
                    # reverse
                    pos = signal
                    entry_price = current_close
                else:
                    pos = 0
                    entry_price = 0.0

        df_bt.iloc[i, df_bt.columns.get_loc('position')] = pos
        df_bt.iloc[i, df_bt.columns.get_loc('pnl')] = cum_pnl
        df_bt.iloc[i, df_bt.columns.get_loc('equity')] = initial_capital + cum_pnl

    return df_bt

# Run the official backtest
backtest_results = backtest_amml(pred_df, threshold_long=0.6, threshold_short=0.4, initial_capital=10_000)

###############################################################################
# 5. PERFORMANCE METRICS
###############################################################################
def calc_sharpe(equity_series: pd.Series, freq_per_year=8760):
    """
    Paper claims an annualized Sharpe ratio using freq_per_year 
    (8760 for hourly bars if continuously collected).
    """
    daily_returns = equity_series.pct_change().dropna()
    avg_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    
    if std_ret == 0:
        return 0
    
    # Annualize
    sharpe = (avg_ret * freq_per_year) / (std_ret * np.sqrt(freq_per_year))
    return sharpe

final_equity = backtest_results['equity'].iloc[-1]
net_profit = final_equity - 10_000
sharpe_ratio = calc_sharpe(backtest_results['equity'], freq_per_year=8760)

# Count trades: each time position changes from 0->1 or 0->-1 or vice versa
trade_changes = backtest_results['position'].diff().fillna(0) != 0
num_trades = trade_changes.sum()

print("\n========== AM-ML BACKTEST RESULTS ==========")
print(f"Final Equity:       ${final_equity:,.2f}")
print(f"Net Profit:         ${net_profit:,.2f}")
print(f"Annualized Sharpe:  {sharpe_ratio:,.2f}")
print(f"Number of Trades:   {num_trades}")

