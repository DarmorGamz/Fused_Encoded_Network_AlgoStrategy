import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from datetime import datetime
from binance.client import Client
from binance.enums import HistoricalKlinesType

###############################################################################
# 1. DATA FETCHING (ALTERNATIVE IMPLEMENTATION)
###############################################################################
def fetch_binance_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    filename: str,
    api_key: str,
    api_secret: str
) -> pd.DataFrame:
    """
    If 'filename' exists locally, load from CSV.
    Otherwise, fetch from Binance (Spot) and save to CSV.
    Returns a DataFrame with columns:
       [open_time, open, high, low, close, volume, close_time, ...]
    """
    if os.path.exists(filename):
        print(f"[INFO] Loading {symbol} data from {filename}")
        df = pd.read_csv(filename, parse_dates=['open_time'])
        return df
    else:
        print(f"[INFO] Downloading {symbol} data from Binance...")
        client = Client(api_key, api_secret)
        raw_klines = client.get_historical_klines(
            symbol,
            interval,
            start_str=start_date,
            end_str=end_date,
            klines_type=HistoricalKlinesType.SPOT
        )
        df = pd.DataFrame(raw_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        # Convert numeric fields
        numeric_cols = ['open','high','low','close','volume',
                        'quote_asset_volume','taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume']
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Convert millisecond timestamps
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Sort and save
        df.sort_values('open_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(filename, index=False)
        return df


###############################################################################
# 2. DATA PREPROCESSING
###############################################################################
def prepare_data(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['datetime', 'price'] indexed by datetime.
    """
    df_proc = df[['open_time', price_col]].copy()
    df_proc.rename(columns={'open_time': 'datetime', price_col: 'price'}, inplace=True)
    df_proc.set_index('datetime', inplace=True)
    return df_proc

def merge_assets(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two dataframes (BTC & ETH) on timestamp, suffixing columns appropriately.
    """
    merged = pd.merge(
        btc_df, eth_df, left_index=True, right_index=True,
        how='inner', suffixes=('_BTC','_ETH')
    )
    merged.dropna(inplace=True)
    merged.sort_index(inplace=True)
    return merged


###############################################################################
# 3. COINTEGRATION CHECKS
###############################################################################
def eg_cointegration_test(series_a: pd.Series, series_b: pd.Series):
    """
    Engle-Granger cointegration test for two price series.
    Returns: (coint_t, p_val, crit_values, hedge_ratio)
    """
    # 1) Hedge ratio from OLS: series_b ~ series_a
    X = sm.add_constant(series_a)
    model = sm.OLS(series_b, X).fit()
    hedge_ratio = model.params.iloc[-1]

    # 2) Perform the Engle-Granger coint test
    coint_t, p_val, crit_vals = coint(series_b, series_a)
    return coint_t, p_val, crit_vals, hedge_ratio


def rolling_cointegration(df: pd.DataFrame,
                          window: int = 200,
                          alpha_level: float = 0.05) -> pd.DataFrame:
    """
    Performs a rolling Engle-Granger cointegration test over a 'window' period.
    Creates 'coint_pval' and 'hedge_ratio' columns in the DataFrame.
    """
    df = df.copy()
    df['coint_pval'] = np.nan
    df['hedge_ratio'] = np.nan

    for i in range(window, len(df)):
        sub_btc = df['price_BTC'].iloc[i-window:i]
        sub_eth = df['price_ETH'].iloc[i-window:i]
        _, pval, _, hr = eg_cointegration_test(sub_btc, sub_eth)
        df.iloc[i, df.columns.get_loc('coint_pval')] = pval
        df.iloc[i, df.columns.get_loc('hedge_ratio')] = hr
    
    return df


###############################################################################
# 4. GENERATE TRADING SIGNALS
###############################################################################
def build_signals(df: pd.DataFrame,
                  lookback: int = 24,
                  z_enter: float = 2.5,
                  z_exit: float = 1.0,
                  coint_win: int = 200,
                  pval_cutoff: float = 0.05) -> pd.DataFrame:
    """
    1) Checks cointegration (p-value < pval_cutoff).
    2) Calculates rolling z-score of the spread = ETH - hedge_ratio*BTC
    3) If z-score > z_enter => short spread
       If z-score < -z_enter => long spread
       Exit if |z-score| < z_exit
    """
    df = rolling_cointegration(df, window=coint_win, alpha_level=pval_cutoff)
    df['spread'] = np.nan
    df['zscore'] = np.nan
    df['signal'] = 0

    for i in range(coint_win, len(df)):
        hr = df['hedge_ratio'].iloc[i]
        # Rolling window for mean/std of the spread
        sub_slice = df.iloc[i - lookback : i]
        spread_vals = sub_slice['price_ETH'] - hr * sub_slice['price_BTC']
        mean_spread = spread_vals.mean()
        std_spread = spread_vals.std() if spread_vals.std() != 0 else 1e-9
        
        current_spread = (df['price_ETH'].iloc[i] - hr * df['price_BTC'].iloc[i])
        zscore = (current_spread - mean_spread) / std_spread
        df.loc[df.index[i], 'spread'] = current_spread
        df.loc[df.index[i], 'zscore'] = zscore

        # Only trade if cointegrated
        if df['coint_pval'].iloc[i] < pval_cutoff:
            if zscore > z_enter:
                df.loc[df.index[i], 'signal'] = -1  # SHORT
            elif zscore < -z_enter:
                df.loc[df.index[i], 'signal'] = 1   # LONG
            else:
                # Potential exit if in a position? We'll handle actual position
                # changes in the backtest logic. Here we just set no "new" signal.
                df.loc[df.index[i], 'signal'] = 0
        else:
            df.loc[df.index[i], 'signal'] = 0

    return df


###############################################################################
# 5. BACKTESTING
###############################################################################
def run_backtest(df: pd.DataFrame,
                 initial_capital: float = 10_000,
                 fee_rate: float = 0.0004,
                 z_exit: float = 1.0,
                 max_adverse: float = 50.0) -> pd.DataFrame:
    """
    Pairs trading backtest:
      - Next-bar execution to avoid look-ahead bias
      - Long or short the spread, exit when |z| < z_exit or a stop is triggered
    """
    data = df.copy().fillna(method='ffill')
    data['position'] = 0
    data['pnl'] = 0.0
    data['equity_curve'] = 0.0

    in_pos = 0
    entry_price_btc = 0.0
    entry_price_eth = 0.0
    entry_spread = 0.0
    cum_pnl = 0.0

    prices_btc = data['price_BTC'].values
    prices_eth = data['price_ETH'].values
    signals = data['signal'].values
    zvals = data['zscore'].values
    hratio = data['hedge_ratio'].values

    for i in range(1, len(data)):
        prev_sig = signals[i-1]
        curr_z = zvals[i]
        hr = hratio[i]
        curr_spread = prices_eth[i] - hr * prices_btc[i]

        if in_pos == 0:
            # Check if a new position should be opened
            if prev_sig == 1:
                # LONG the spread => long ETH, short BTC
                in_pos = 1
                entry_price_eth = prices_eth[i]
                entry_price_btc = prices_btc[i]
                entry_spread = curr_spread
            elif prev_sig == -1:
                # SHORT the spread => short ETH, long BTC
                in_pos = -1
                entry_price_eth = prices_eth[i]
                entry_price_btc = prices_btc[i]
                entry_spread = curr_spread
        else:
            # We have a position, check exit or stop
            if in_pos == 1:
                # LONG spread: if the spread moves adversely or if |z| < z_exit
                adverse = entry_spread - curr_spread
                if adverse > max_adverse or abs(curr_z) < z_exit:
                    exit_eth = prices_eth[i]
                    exit_btc = prices_btc[i]
                    # PnL = (exit_eth - entry_eth) + (entry_btc - exit_btc)
                    trade_pnl = (exit_eth - entry_price_eth) + (entry_price_btc - exit_btc)
                    trade_pnl -= fee_rate * abs(trade_pnl)
                    cum_pnl += trade_pnl
                    in_pos = 0
            else:
                # SHORT spread
                adverse = curr_spread - entry_spread
                if adverse > max_adverse or abs(curr_z) < z_exit:
                    exit_eth = prices_eth[i]
                    exit_btc = prices_btc[i]
                    # PnL = (entry_eth - exit_eth) + (exit_btc - entry_btc)
                    trade_pnl = (entry_price_eth - exit_eth) + (exit_btc - entry_price_btc)
                    trade_pnl -= fee_rate * abs(trade_pnl)
                    cum_pnl += trade_pnl
                    in_pos = 0

        data.at[data.index[i], 'position'] = in_pos
        data.at[data.index[i], 'pnl'] = cum_pnl
    
    data['equity_curve'] = data['pnl'] + initial_capital
    return data


###############################################################################
# 6. PERFORMANCE METRICS
###############################################################################
def compute_sharpe(equity_series: pd.Series, freq_hours=1) -> float:
    """
    Computes an annualized Sharpe ratio given an equity curve.
    - freq_hours: number of hours per bar. (1 for 1-hour bars)
    """
    rets = equity_series.pct_change().dropna()
    avg_ret = rets.mean()
    std_ret = rets.std()

    if std_ret == 0:
        return 0.0

    # Approx 8760 hours in a year => annualize
    annual_factor = 8760 / freq_hours
    sharpe = (avg_ret * annual_factor) / (std_ret * np.sqrt(annual_factor))
    return sharpe


###############################################################################
# MAIN EXECUTION EXAMPLE
###############################################################################
if __name__ == "__main__":

    # Provide your own API keys or use existing CSV files.
    API_KEY = "<YOUR_BINANCE_API_KEY>"
    API_SECRET = "<YOUR_BINANCE_API_SECRET>"

    # 1) Fetch or load BTC data
    btc_raw = fetch_binance_data(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_date="1 Jan, 2021",
        end_date=None,
        filename="BTCUSDT_1H.csv",
    )
    # 2) Fetch or load ETH data
    eth_raw = fetch_binance_data(
        symbol="ETHUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_date="1 Jan, 2021",
        end_date=None,
        filename="ETHUSDT_1H.csv",
    )

    # Preprocess
    btc_proc = prepare_data(btc_raw, price_col='close')
    eth_proc = prepare_data(eth_raw, price_col='close')

    # Merge
    pair_data = merge_assets(btc_proc, eth_proc)

    # Generate signals
    signals_df = build_signals(
        pair_data,
        lookback=24,
        z_enter=2.5,
        z_exit=1.0,
        coint_win=200,
        pval_cutoff=0.05
    )

    # Backtest
    bt_result = run_backtest(
        signals_df,
        initial_capital=10_000,
        fee_rate=0.0004,
        z_exit=1.0,
        max_adverse=50.0
    )

    # Performance
    final_eq = bt_result['equity_curve'].iloc[-1]
    net_pnl = final_eq - 10_000
    sr = compute_sharpe(bt_result['equity_curve'], freq_hours=1)

    # Count trades => # times position changes from 0 to ±1 or vice versa
    pos_diff = bt_result['position'].diff().fillna(0)
    # A trade is whenever pos changes from 0 to ±1 or ±1 back to 0
    # but typically we count only 0->1 or 0->-1 as "trade entries"
    num_trades = ((pos_diff != 0) & (bt_result['position'] != 0)).sum()

    print("\n================ BACKTEST RESULTS ================")
    print(f"Final Equity:       ${final_eq:,.2f}")
    print(f"Net PnL:            ${net_pnl:,.2f}")
    print(f"Annualized Sharpe:  {sr:,.2f}")
    print(f"Number of Trades:   {num_trades}")
