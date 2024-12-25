import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from datetime import datetime
from binance.client import Client
from binance.enums import HistoricalKlinesType


###############################################################################
# 1. DATA ACQUISITION
###############################################################################
API_KEY = "<YOUR_BINANCE_API_KEY>"
API_SECRET = "<YOUR_BINANCE_API_SECRET>"
client = Client(API_KEY, API_SECRET)


def fetch_klines_if_needed(symbol: str,
                           interval: str = Client.KLINE_INTERVAL_1HOUR,
                           start_date: str = "1 Jan, 2021",
                           end_date: str = None,
                           filename: str = None) -> pd.DataFrame:
    """
    Fetches historical klines (candles) from Binance if 'filename' doesn't exist locally.
    Saves to CSV and returns as a pandas DataFrame. Otherwise, loads local CSV with parse_dates.
    """
    if filename is None:
        filename = f"{symbol}_{interval}_data.csv"

    if os.path.exists(filename):
        print(f"[INFO] Loading data locally from {filename}...")
        # Since the CSV should have normal date strings, parse them as standard datetimes:
        df = pd.read_csv(filename, parse_dates=['open_time'])
        return df
    else:
        print(f"[INFO] Fetching data from Binance for {symbol}...")
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_date,
            end_str=end_date,
            klines_type=HistoricalKlinesType.SPOT
        )
        # Convert to DataFrame
        df = pd.DataFrame(klines, 
                          columns=['open_time', 'open', 'high', 'low', 'close',
                                   'volume', 'close_time', 'quote_asset_volume',
                                   'number_of_trades', 'taker_buy_base_asset_volume',
                                   'taker_buy_quote_asset_volume', 'ignore'])
        
        # Convert numeric columns
        numeric_cols = ['open','high','low','close','volume','quote_asset_volume',
                        'taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert 'open_time' from ms to a normal datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Sort by time
        df.sort_values('open_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save to CSV. 'open_time' now has ISO8601 date strings:
        df.to_csv(filename, index=False)
        return df


# Fetch BTC/USDT data
btc_df = fetch_klines_if_needed(symbol="BTCUSDT",
                                interval=Client.KLINE_INTERVAL_1HOUR,
                                start_date="1 Jan, 2021",
                                filename="BTCUSDT_1H.csv")

# Fetch ETH/USDT data
eth_df = fetch_klines_if_needed(symbol="ETHUSDT",
                                interval=Client.KLINE_INTERVAL_1HOUR,
                                start_date="1 Jan, 2021",
                                filename="ETHUSDT_1H.csv")


###############################################################################
# 2. DATA PREPROCESSING
###############################################################################
def preprocess_data(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Prepares kline data into a simpler format with columns [datetime, price].
    Ensures no missing timestamps (if so, consider forward fill or drop).
    """
    df = df[['open_time', price_col]].copy()
    df.rename(columns={'open_time': 'datetime',
                       price_col: 'price'}, inplace=True)
    df.set_index('datetime', inplace=True)
    return df

btc_df_prep = preprocess_data(btc_df)
eth_df_prep = preprocess_data(eth_df)

# Merge on the index (datetime)
merged_df = pd.merge(btc_df_prep, eth_df_prep, left_index=True, right_index=True,
                     how='inner', suffixes=('_BTC', '_ETH'))

merged_df.dropna(inplace=True)  # Drop any rows with NaN
merged_df.sort_index(inplace=True)


###############################################################################
# 3. COINTEGRATION UTILITIES
###############################################################################
def engle_granger_cointegration(x: pd.Series, y: pd.Series):
    """
    Performs Engle-Granger cointegration test on x and y, returning:
        - coint_t (test statistic)
        - p_value
        - critical_values
        - hedge_ratio (from linear regression y ~ x)
    """
    # 1) Regress y on x to find hedge ratio
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    hedge_ratio = model.params.iloc[-1]  # last parameter is slope

    # 2) Engle-Granger cointegration test
    coint_t, p_value, crit_values = coint(y, x)
    return coint_t, p_value, crit_values, hedge_ratio


def rolling_cointegration_check(df: pd.DataFrame,
                                window: int = 200,
                                alpha: float = 0.05) -> pd.DataFrame:
    """
    Checks cointegration on a rolling basis for the last 'window' bars.
    Creates columns:
      - coint_pval: The Engle-Granger p-value in that window
      - hedge_ratio: The slope in that window
    We only do this once we have at least 'window' data points.
    """
    df = df.copy()
    df['coint_pval'] = np.nan
    df['hedge_ratio'] = np.nan
    
    for i in range(window, len(df)):
        sub_btc = df['price_BTC'].iloc[i - window:i]
        sub_eth = df['price_ETH'].iloc[i - window:i]
        _, p_val, _, hr = engle_granger_cointegration(sub_btc, sub_eth)
        df.loc[df.index[i], 'coint_pval'] = p_val
        df.loc[df.index[i], 'hedge_ratio'] = hr
    
    return df


###############################################################################
# 4. SIGNAL GENERATION (Z-SCORE) WITH ROLLING COINTEGRATION FILTER
###############################################################################
def generate_signals(merged: pd.DataFrame,
                     lookback: int = 24, 
                     entry_z: float = 2.5,   # more conservative
                     exit_z: float = 1.0,    # likewise, give room to revert
                     coint_window: int = 200,
                     coint_alpha: float = 0.05):
    """
    Generates trading signals based on:
      1) Rolling cointegration test (p-value < coint_alpha)
      2) Z-score of the spread = ETH - hedge_ratio * BTC
         with a rolling lookback window for mean & std.
    We only enter trades if p-value < coint_alpha.
    """
    df = rolling_cointegration_check(merged, window=coint_window, alpha=coint_alpha)
    
    # Initialize columns for signals
    df['spread'] = np.nan
    df['spread_mean'] = np.nan
    df['spread_std'] = np.nan
    df['zscore'] = np.nan
    df['signal'] = 0
    
    # Compute spread using the rolling hedge ratio
    valid_idx = df.index[df['hedge_ratio'].notna()]
    
    for i in range(coint_window, len(df)):
        hr = df.loc[df.index[i], 'hedge_ratio']
        # We'll look back 'lookback' bars to compute rolling mean/std
        window_slice = df.iloc[i - lookback + 1 : i + 1]
        
        spread_series = window_slice['price_ETH'] - hr * window_slice['price_BTC']
        spread_mean = spread_series.mean()
        spread_std = spread_series.std()
        
        current_spread = spread_series.iloc[-1]
        zscore_value = (current_spread - spread_mean) / spread_std if spread_std != 0 else 0
        
        df.loc[df.index[i], 'spread'] = current_spread
        df.loc[df.index[i], 'spread_mean'] = spread_mean
        df.loc[df.index[i], 'spread_std'] = spread_std
        df.loc[df.index[i], 'zscore'] = zscore_value
        
        # Only generate signals if p-value < coint_alpha
        if df.loc[df.index[i], 'coint_pval'] < coint_alpha:
            # Enter Long if zscore < -entry_z
            if zscore_value < -entry_z:
                df.loc[df.index[i], 'signal'] = 1
            # Enter Short if zscore > entry_z
            elif zscore_value > entry_z:
                df.loc[df.index[i], 'signal'] = -1
            else:
                df.loc[df.index[i], 'signal'] = 0
        else:
            # If not cointegrated, no new positions
            df.loc[df.index[i], 'signal'] = 0
    
    return df

signal_df = generate_signals(merged_df, 
                             lookback=24,    # or larger if you want a smoother signal
                             entry_z=2.5,    # can tweak
                             exit_z=1.0, 
                             coint_window=200,
                             coint_alpha=0.05)


###############################################################################
# 5. BACKTEST (AVOIDING LOOK-AHEAD BIAS) + STOP LOSS
###############################################################################
def backtest_pairs(signal_data: pd.DataFrame,
                   initial_capital: float = 10000,
                   trading_fee_rate: float = 0.0004,
                   exit_z: float = 1.0,
                   max_adverse_move: float = 50.0):
    """
    Simplified backtest for cointegration-based pairs trading:
      - Next-bar execution to avoid look-ahead bias.
      - Position can be long spread (1) or short spread (-1) or flat (0).
      - We exit if |zscore| < exit_z OR if the trade hits a stop loss (max_adverse_move).
        * 'max_adverse_move' is a fixed threshold in spread terms.

    Returns a DataFrame with 'position', 'pnl', 'equity_curve'.
    """
    df = signal_data.copy().fillna(method='ffill')  # forward-fill hedge_ratio, etc., if needed
    
    df['position'] = 0
    df['pnl'] = 0.0
    df['equity_curve'] = 0.0
    
    in_position = 0
    entry_price_btc = 0
    entry_price_eth = 0
    entry_spread = 0
    total_pnl = 0.0
    
    prices_btc = df['price_BTC'].values
    prices_eth = df['price_ETH'].values
    signals = df['signal'].values
    zscores = df['zscore'].values
    hedge_ratios = df['hedge_ratio'].values
    
    for i in range(1, len(df)):
        prev_signal = signals[i-1]
        current_z = zscores[i]
        hr = hedge_ratios[i]
        
        # Calculate the "current" spread for stop-loss checks
        current_spread = prices_eth[i] - hr * prices_btc[i]
        
        if in_position == 0:
            # Currently flat, decide to enter on the previous bar's signal
            if prev_signal == 1:
                # LONG spread => long ETH, short BTC
                in_position = 1
                entry_price_eth = prices_eth[i]
                entry_price_btc = prices_btc[i]
                entry_spread = current_spread
            elif prev_signal == -1:
                # SHORT spread => short ETH, long BTC
                in_position = -1
                entry_price_eth = prices_eth[i]
                entry_price_btc = prices_btc[i]
                entry_spread = current_spread

        else:
            # Already in a position
            if in_position == 1:
                # LONG spread => exit if abs(zscore) < exit_z or if stop loss triggers
                # Stop loss: if the spread moves 'adversely' more than max_adverse_move
                #   adverse_move = entry_spread - current_spread
                adverse_move = entry_spread - current_spread
                if adverse_move > max_adverse_move:
                    # stop loss triggered
                    exit_price_eth = prices_eth[i]
                    exit_price_btc = prices_btc[i]
                    eth_pnl = exit_price_eth - entry_price_eth
                    btc_pnl = entry_price_btc - exit_price_btc
                    trade_pnl = eth_pnl + btc_pnl
                    trade_pnl_after_fees = trade_pnl - trading_fee_rate * abs(trade_pnl)
                    total_pnl += trade_pnl_after_fees
                    in_position = 0
                else:
                    if abs(current_z) < exit_z:
                        # Normal exit
                        exit_price_eth = prices_eth[i]
                        exit_price_btc = prices_btc[i]
                        eth_pnl = exit_price_eth - entry_price_eth
                        btc_pnl = entry_price_btc - exit_price_btc
                        trade_pnl = eth_pnl + btc_pnl
                        trade_pnl_after_fees = trade_pnl - trading_fee_rate * abs(trade_pnl)
                        total_pnl += trade_pnl_after_fees
                        in_position = 0

            else:
                # SHORT spread => exit if abs(zscore) < exit_z or if stop loss triggers
                # Stop loss: if the spread moves 'adversely' more than max_adverse_move
                #   adverse_move = current_spread - entry_spread
                adverse_move = current_spread - entry_spread
                if adverse_move > max_adverse_move:
                    # stop loss triggered
                    exit_price_eth = prices_eth[i]
                    exit_price_btc = prices_btc[i]
                    eth_pnl = entry_price_eth - exit_price_eth  # short ETH => (entry - exit)
                    btc_pnl = exit_price_btc - entry_price_btc  # long BTC => (exit - entry)
                    trade_pnl = eth_pnl + btc_pnl
                    trade_pnl_after_fees = trade_pnl - trading_fee_rate * abs(trade_pnl)
                    total_pnl += trade_pnl_after_fees
                    in_position = 0
                else:
                    if abs(current_z) < exit_z:
                        exit_price_eth = prices_eth[i]
                        exit_price_btc = prices_btc[i]
                        eth_pnl = entry_price_eth - exit_price_eth
                        btc_pnl = exit_price_btc - entry_price_btc
                        trade_pnl = eth_pnl + btc_pnl
                        trade_pnl_after_fees = trade_pnl - trading_fee_rate * abs(trade_pnl)
                        total_pnl += trade_pnl_after_fees
                        in_position = 0

        df.at[df.index[i], 'position'] = in_position
        df.at[df.index[i], 'pnl'] = total_pnl
    
    df['equity_curve'] = df['pnl'] + initial_capital
    return df


# Perform the backtest
backtest_df = backtest_pairs(signal_df,
                             initial_capital=10000,
                             trading_fee_rate=0.0004,
                             exit_z=1.0,
                             max_adverse_move=50.0)


###############################################################################
# 6. RESULT EVALUATION
###############################################################################
def calculate_sharpe(equity_series: pd.Series, freq_hours=1):
    """
    Calculates a simple Sharpe ratio for the equity curve.
    :param equity_series: The equity curve series over time
    :param freq_hours: The frequency of each bar (1 for hourly, 24 for daily, etc.)
    :return: Sharpe ratio
    """
    returns = equity_series.pct_change().dropna()
    # Convert these hourly returns to an annualized figure:
    # Approx 24 * 365 = 8760 hours/year, so for hourly data:
    #   annualized_return = avg_return_per_hour * 8760
    #   annualized_vol    = std_return_per_hour * sqrt(8760)
    avg_ret = returns.mean()
    std_ret = returns.std()
    
    annual_factor = 8760 / freq_hours  # approx hours in a year
    
    if std_ret == 0:
        return 0
    
    sharpe = (avg_ret * annual_factor) / (std_ret * np.sqrt(annual_factor))
    return sharpe


# Final performance metrics
final_equity = backtest_df['equity_curve'].iloc[-1]
net_profit = final_equity - 10000
sharpe_ratio = calculate_sharpe(backtest_df['equity_curve'], freq_hours=1)

print("\n[RESULTS]")
print(f"Final Equity: ${final_equity:.2f}")
print(f"Net Profit: ${net_profit:.2f}")
print(f"Estimated Sharpe Ratio: {sharpe_ratio:.2f}")

# Count the number of trades: (0->±1 or ±1->0 transitions)
trades_mask = (backtest_df['position'].diff() != 0) & (backtest_df['position'] != 0)
num_trades = trades_mask.sum()
print(f"Number of trades executed: {num_trades}")
