# Individual Stock Metrics (based on log returns)

import yfinance as yf
import numpy as np
import pandas as pd

def get_close_data(ticker, start, end):
    data = yf.download(ticker, start, end, multi_level_index = False)
    close_data = data.Close.to_frame()
    return close_data

def log_daily_returns(close_data):
    return np.log(close_data.div(close_data.shift(1)).dropna())

def cumulative_returns(log_daily_ret):
    return log_daily_ret.cumsum()
 
def log_annualized_returns(log_daily_ret):
    return log_daily_ret.mean() * 252

def annualized_volatility(log_daily_ret):
    return log_daily_ret.std() * np.sqrt(252)

def raw_sharpe(log_annl_ret, annl_vol):
    return log_annl_ret/annl_vol

def sharpe_ratio(log_annl_ret, annl_vol, rf_rate):
    # can use 0.03 as placeholder for rf_rate
    return (log_annl_ret - np.log(1 + rf_rate))/annl_vol

def log_metrics(ticker, start, end):
    close_data = get_close_data(ticker, start, end)
    log_daily_ret = log_daily_returns(close_data)
    cum_ret = np.exp(cumulative_returns(log_daily_ret))
    total_return = float(cum_ret.iloc[-1])
    log_annl_ret = float(np.exp(log_annualized_returns(log_daily_ret)) - 1)
    annl_vol = float(annualized_volatility(log_daily_ret))
    raw_sharpe_ratio = raw_sharpe(log_annl_ret, annl_vol)
    sharpe = sharpe_ratio(log_annl_ret, annl_vol, 0.03)

    order = [
    "Log Cumulative Return",
    "Log Annualized Return",
    "Log Annualized Volatility",
    "Raw Sharpe Ratio",
    "Sharpe Ratio"
    ]

    metrics =  {
        "Log Cumulative Return": total_return,
        "Log Annualized Return": log_annl_ret,
        "Log Annualized Volatility": annl_vol,
        "Raw Sharpe Ratio": raw_sharpe_ratio,
        "Sharpe Ratio": sharpe
    }

    return pd.DataFrame(metrics, index = [ticker])[order]