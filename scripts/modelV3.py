import os
import numpy as np
import pandas as pd
import xgboost as xgb
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

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
START_DATE = "1 Jan, 2021"
FILENAME = "BTCUSDT_AMML_1H.csv"

def fetch_data(symbol: str = SYMBOL, 
               interval: str = INTERVAL, 
               start: str = START_DATE,
               filename: str = FILENAME) -> pd.DataFrame:
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

raw_df = fetch_data()

###############################################################################
# 2. FEATURE ENGINEERING
###############################################################################
def build_features(df: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat['returns'] = df_feat['close'].pct_change()
    df_feat['rolling_ret'] = df_feat['close'].pct_change(lookback)

    df_feat['rolling_vol'] = df_feat['returns'].rolling(lookback).std()
    df_feat['rolling_vol'] = df_feat['rolling_vol'].bfill()

    df_feat['rolling_volume'] = df_feat['volume'].rolling(lookback).mean()
    df_feat['rolling_volume'] = df_feat['rolling_volume'].bfill()

    df_feat['future_close'] = df_feat['close'].shift(-1)
    df_feat['target'] = (df_feat['future_close'] > df_feat['close']).astype(int)
    df_feat.dropna(inplace=True)
    feature_cols = ['returns', 'rolling_ret', 'rolling_vol', 'rolling_volume']
    df_feat = df_feat[['open_time', 'close', 'target'] + feature_cols].copy()
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
    df_out = df.copy()
    df_out['prediction_prob'] = np.nan
    initial_train_len = int(0.2 * len(df))  # Use 20% of data for initial training
    if len(df_out) < initial_train_len:
        raise ValueError("Not enough data for initial training window.")
    X_all = df_out[feature_cols].values
    y_all = df_out['target'].values
    start_idx = initial_train_len
    while start_idx < len(df_out) - test_size:
        X_train = X_all[:start_idx]
        y_train = y_all[:start_idx]
        end_idx = min(start_idx + test_size, len(df_out))
        X_test = X_all[start_idx:end_idx]
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        test_probs = model.predict_proba(X_test)[:,1]
        df_out.iloc[start_idx:end_idx, df_out.columns.get_loc('prediction_prob')] = test_probs
        start_idx += test_size
    return df_out

feature_list = ['returns', 'rolling_ret', 'rolling_vol', 'rolling_volume']
pred_df = walk_forward_prediction(feat_df, feature_cols=feature_list)

###############################################################################
# 4. BACKTEST
###############################################################################
def backtest_amml(df: pd.DataFrame,
                  threshold_long: float = 0.6,
                  threshold_short: float = 0.4,
                  initial_capital: float = 10_000,
                  fee_rate: float = 0):
    df_bt = df.copy()
    df_bt['position'] = 0
    df_bt['pnl'] = 0.0
    df_bt['equity'] = float(initial_capital)  # Ensure 'equity' is initialized as float

    pos = 0  # Current position: 1 (long), -1 (short), 0 (flat)
    entry_price = 0.0
    cum_pnl = 0.0

    close_prices = df_bt['close'].values
    probs = df_bt['prediction_prob'].fillna(0).values  # Ensure no NaN in prediction_prob

    for i in range(1, len(df_bt)):
        prev_prob = probs[i-1]  # Use previous bar's signal
        current_close = close_prices[i]

        # Determine signal
        if prev_prob > threshold_long:
            signal = 1
        elif prev_prob < threshold_short:
            signal = -1
        else:
            signal = 0

        # Check if we need to enter, exit, or reverse
        if pos == 0:
            if signal != 0:
                pos = signal
                entry_price = current_close
        else:
            if signal == 0 or signal == -pos:
                exit_price = current_close
                trade_pnl = (exit_price - entry_price) * pos
                fee = fee_rate * abs(exit_price) * 2
                cum_pnl += trade_pnl - fee

                # Reverse position if signal changes
                if signal == -pos:
                    pos = signal
                    entry_price = current_close
                else:
                    pos = 0
                    entry_price = 0.0

        df_bt.iloc[i, df_bt.columns.get_loc('position')] = pos
        df_bt.iloc[i, df_bt.columns.get_loc('pnl')] = cum_pnl
        df_bt.iloc[i, df_bt.columns.get_loc('equity')] = float(initial_capital + cum_pnl)  # Explicitly cast to float

    return df_bt


backtest_results = backtest_amml(pred_df, threshold_long=0.6, threshold_short=0.4, initial_capital=10_000)

###############################################################################
# 5. PERFORMANCE METRICS
###############################################################################
def calc_sharpe(equity_series: pd.Series, freq_per_year=8760):
    if equity_series.isnull().any():
        print("[WARNING] Equity series contains NaN values.")
        return np.nan
    daily_returns = equity_series.pct_change().dropna()
    avg_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    if std_ret == 0:
        print("[WARNING] Standard deviation is zero. Sharpe ratio is undefined.")
        return np.nan
    sharpe = (avg_ret * freq_per_year) / (std_ret * np.sqrt(freq_per_year))
    return sharpe

final_equity = backtest_results['equity'].iloc[-1]
net_profit = final_equity - 10_000
sharpe_ratio = calc_sharpe(backtest_results['equity'], freq_per_year=8760)
trade_changes = backtest_results['position'].diff().fillna(0) != 0
num_trades = trade_changes.sum()

print("\n========== AM-ML BACKTEST RESULTS ==========")
print(f"Final Equity:       ${final_equity:,.2f}")
print(f"Net Profit:         ${net_profit:,.2f}")
print(f"Annualized Sharpe:  {sharpe_ratio:,.2f}")
print(f"Number of Trades:   {num_trades}")
