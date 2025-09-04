# Individual Stock Metrics (based on simple arithmetic returns)

import yfinance as yf
import numpy as np
import pandas as pd

def get_close_data(ticker, start, end):
    data = yf.download(ticker, start, end, multi_level_index = False)
    close_data = data.Close.to_frame()
    return close_data

def simple_daily_returns(close_data):
    return close_data.pct_change().dropna()

def cumulative_returns(simp_daily_ret):
    return (simp_daily_ret + 1).cumprod()
 
def simple_annualized_returns(simp_daily_ret):
    return float(simp_daily_ret.mean() * 252)

def cagr(cum_returns):
    final_value = cum_returns.iloc[-1]
    initial_value = cum_returns.iloc[0]
    growth_factor = final_value/initial_value
    trading_days = len(cum_returns)
    years = trading_days/252
    cagr = growth_factor ** (1 / years) - 1
    return float(cagr)

def annualized_volatility(simp_daily_ret):
    return float(simp_daily_ret.std() * np.sqrt(252))

def max_drawdown(cum_returns):
    cum_returns['cummax'] = cum_returns.cummax()
    cum_returns['drawdowns'] = (cum_returns['cummax'] - cum_returns.iloc[:, 0])/cum_returns['cummax']
    return cum_returns['drawdowns'].max()

def raw_sharpe(simp_annl_ret, annl_vol):
    return float(simp_annl_ret/annl_vol)

def sharpe_ratio(simp_annl_ret, annl_vol, rf_rate):
    # can use 0.03 as placeholder for rf_rate
    return float((simp_annl_ret - rf_rate)/annl_vol)


def simple_metrics(ticker, start, end):
    close_data = get_close_data(ticker, start, end)
    simp_daily_ret = simple_daily_returns(close_data)
    cum_ret = cumulative_returns(simp_daily_ret)
    total_return = float(cum_ret.iloc[-1])
    simp_annl_ret = simple_annualized_returns(simp_daily_ret)
    cagr_value = cagr(cum_ret)
    annl_vol = annualized_volatility(simp_daily_ret)
    max_draw = max_drawdown(cum_ret)
    raw_sharpe_ratio = raw_sharpe(simp_annl_ret, annl_vol)
    sharpe = sharpe_ratio(simp_annl_ret, annl_vol, 0.03)

    order = [
    "Cumulative Return",
    "Simple Annualized Return",
    "CAGR",
    "Simple Annualized Volatility",
    "Maximum Drawdown",
    "Raw Sharpe Ratio",
    "Sharpe Ratio"
    ]

    metrics =  {
        "Cumulative Return": total_return,
        "Simple Annualized Return": simp_annl_ret,
        "CAGR": cagr_value,
        "Simple Annualized Volatility": annl_vol,
        "Maximum Drawdown": max_draw,
        "Raw Sharpe Ratio": raw_sharpe_ratio,
        "Sharpe Ratio": sharpe
    }

    return pd.DataFrame(metrics, index = [ticker])[order]
