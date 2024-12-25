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
client = Client()

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
        numeric_cols = [
            'open','high','low','close','volume','quote_asset_volume',
            'taker_buy_base_asset_volume','taker_buy_quote_asset_volume'
        ]
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
# 3. COINTEGRATION TEST
###############################################################################
def find_cointegration(x: pd.Series, y: pd.Series, alpha=0.05):
    """
    Performs Engle-Granger cointegration test on x and y.
    Returns:
        - coint_t (test statistic)
        - p_value
        - critical_values
        - hedge_ratio (from linear regression y ~ x)
    """
    # 1) Regress y on x to find hedge ratio
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    # Use .iloc[-1] to get the slope (hedge ratio) safely:
    hedge_ratio = model.params.iloc[-1]  # last parameter is slope

    # 2) Engle-Granger cointegration test on (y, x)
    coint_t, p_value, crit_values = coint(y, x)  # pass as positional arguments

    return coint_t, p_value, crit_values, hedge_ratio

btc_prices = merged_df['price_BTC']
eth_prices = merged_df['price_ETH']

coint_t, p_val, crit_val, hedge_ratio = find_cointegration(btc_prices, eth_prices)

print("[INFO] Cointegration test results:")
print(f"  t-statistic: {coint_t:.3f}")
print(f"  p-value: {p_val:.5f}")
print(f"  critical values: {crit_val}")
print(f"  hedge ratio (ETH ~ BTC): {hedge_ratio:.4f}")

###############################################################################
# 4. SIGNAL GENERATION (Z-SCORE)
###############################################################################
def generate_signals(merged: pd.DataFrame, hedge_ratio: float,
                     lookback: int = 24, entry_z: float = 2.0, exit_z: float = 0.5):
    """
    Generates trading signals based on z-score of the spread:
      spread = ETH_price - hedge_ratio * BTC_price
    Then compute rolling mean & std to get z-score.
    
    :param merged: DataFrame with columns [price_BTC, price_ETH]
    :param hedge_ratio: from cointegration regression
    :param lookback: rolling window size for z-score
    :param entry_z: absolute z-score threshold to enter trades
    :param exit_z: z-score threshold to exit trades
    :return: DataFrame with columns [spread, zscore, signal]
             signal = 1 (long spread), -1 (short spread), 0 (flat)
    """
    df = merged.copy()
    df['spread'] = df['price_ETH'] - hedge_ratio * df['price_BTC']
    df['spread_mean'] = df['spread'].rolling(lookback).mean()
    df['spread_std'] = df['spread'].rolling(lookback).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # Initialize with no position
    df['signal'] = 0
    
    # We will set signals in a vectorized way, then handle position logic in the backtest
    # This is just the "intended" position, but the actual position in backtest can differ
    # due to exit triggers.
    
    # Enter Long Spread if zscore < -entry_z
    df.loc[df['zscore'] < -entry_z, 'signal'] = 1
    # Enter Short Spread if zscore > +entry_z
    df.loc[df['zscore'] > entry_z, 'signal'] = -1
    # Everything else remains 0 in terms of "entry intention"
    # Actual position management is in the backtest
    
    return df

signal_df = generate_signals(merged_df, hedge_ratio)

###############################################################################
# 5. BACKTEST (AVOIDING LOOK-AHEAD BIAS)
###############################################################################
def backtest_pairs(signal_data: pd.DataFrame,
                   hedge_ratio: float,
                   entry_z: float = 2.0,
                   exit_z: float = 0.5,
                   initial_capital: float = 10000,
                   trading_fee_rate: float = 0.0004):
    """
    Simplified backtest for cointegration-based pairs trading:
    - We use next-bar execution to avoid look-ahead bias.
    - Position can be long spread (1) or short spread (-1) or flat (0).
    - We exit if z-score crosses the 'exit_z' boundary towards 0, or if cointegration might break.
    
    Returns a DataFrame with columns:
      'position': actual position (-1, 0, +1)
      'pnl': incremental PnL from the trade
      'equity_curve': cumulative PnL
    """
    df = signal_data.copy()
    
    # Actual position that the strategy holds at each bar
    df['position'] = 0
    
    in_position = 0  # current position state (-1, 0, +1)
    entry_price_btc = 0
    entry_price_eth = 0
    
    # We'll accumulate PnL over time
    df['pnl'] = 0.0
    df['equity_curve'] = 0.0
    
    # We assume equal capital is used for each trade. This is a simplified approach.
    # In reality, you might size positions based on volatility or other factors.
    
    # Keep track of total PnL in USDT terms
    total_pnl = 0.0
    
    prices_btc = df['price_BTC'].values
    prices_eth = df['price_ETH'].values
    zscores = df['zscore'].values
    signals = df['signal'].values
    spread = df['spread'].values
    
    for i in range(1, len(df)):
        # "Look-back" to the signal from the previous bar to decide today's action
        prev_signal = signals[i-1]  # from previous bar
        current_z = zscores[i]      # current bar z-score
        
        if in_position == 0:
            # Currently flat, check if we should enter
            if prev_signal == 1:
                # Go LONG the spread:
                #   => Long ETH, Short BTC
                in_position = 1
                entry_price_eth = prices_eth[i]  # next-bar open (approximated by i's price)
                entry_price_btc = prices_btc[i]
            elif prev_signal == -1:
                # Go SHORT the spread:
                #   => Short ETH, Long BTC
                in_position = -1
                entry_price_eth = prices_eth[i]
                entry_price_btc = prices_btc[i]
        else:
            # We are in a position, check if we need to exit
            if in_position == 1:
                # We are LONG spread
                # exit if zscore crosses 'exit_z' back upwards or passes 0
                if abs(current_z) < exit_z:
                    # Exit position
                    exit_price_eth = prices_eth[i]
                    exit_price_btc = prices_btc[i]
                    
                    # PnL calculation in USDT
                    #   Long ETH PnL: (exit_price_eth - entry_price_eth)
                    #   Short BTC PnL: (entry_price_btc - exit_price_btc)
                    eth_pnl = exit_price_eth - entry_price_eth
                    btc_pnl = entry_price_btc - exit_price_btc
                    trade_pnl = eth_pnl + btc_pnl
                    
                    # Subtract trading fee approximation:
                    trade_pnl_after_fees = trade_pnl - trading_fee_rate * abs(trade_pnl)
                    
                    total_pnl += trade_pnl_after_fees
                    in_position = 0  # flat
            else:
                # We are SHORT spread
                # exit if zscore crosses 'exit_z' or passes 0
                if abs(current_z) < exit_z:
                    # Exit position
                    exit_price_eth = prices_eth[i]
                    exit_price_btc = prices_btc[i]
                    
                    # PnL calculation
                    #   Short ETH PnL: (entry_price_eth - exit_price_eth)
                    #   Long BTC PnL: (exit_price_btc - entry_price_btc)
                    eth_pnl = entry_price_eth - exit_price_eth
                    btc_pnl = exit_price_btc - entry_price_btc
                    trade_pnl = eth_pnl + btc_pnl
                    
                    # Subtract trading fee
                    trade_pnl_after_fees = trade_pnl - trading_fee_rate * abs(trade_pnl)
                    
                    total_pnl += trade_pnl_after_fees
                    in_position = 0  # flat
        
        # Assign the position to the DataFrame row for the current bar
        df.at[df.index[i], 'position'] = in_position
        df.at[df.index[i], 'pnl'] = total_pnl
    
    df['equity_curve'] = df['pnl'] + initial_capital
    return df

backtest_df = backtest_pairs(signal_df, hedge_ratio)

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
    
    annual_factor = 8760 / freq_hours  # number of hours in a year (approx)
    
    if std_ret == 0:
        return 0
    
    sharpe = (avg_ret * annual_factor) / (std_ret * np.sqrt(annual_factor))
    return sharpe

final_equity = backtest_df['equity_curve'].iloc[-1]
net_profit = final_equity - 10000
sharpe_ratio = calculate_sharpe(backtest_df['equity_curve'], freq_hours=1)

print("\n[RESULTS]")
print(f"Final Equity: ${final_equity:.2f}")
print(f"Net Profit: ${net_profit:.2f}")
print(f"Estimated Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Number of trades executed: {((backtest_df['position'].diff()!=0)&(backtest_df['position']!=0)).sum()}")
