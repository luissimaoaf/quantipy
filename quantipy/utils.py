import pandas as pd
import numpy as np
import datetime
from itertools import product


def cum_return(equity):
    return equity[-1]/equity[0] - 1


def compute_drawdown(equity):
    
    rolling_max = equity.cummax()
    tick_dd = equity/rolling_max - 1.0
    max_dd = tick_dd.cummin()
    
    return [tick_dd, max_dd]


def compute_rolling_drawdown(equity, window):
    
    rolling_max = equity.rolling(window, min_periods=1).max()
    tick_dd = equity/rolling_max - 1.0
    max_dd = tick_dd.rolling(window, min_periods=1).min()
    
    return [tick_dd, max_dd]


def compute_drawdown_length(dd):
    
    drawdown_lengths = []
    length = 0
    for i in range(len(dd)):
        if dd[i] < 0:
            length +=1
        elif length > 0:
            drawdown_lengths.append(length)
            length = 0
    
    if drawdown_lengths == [] and length > 0:
        drawdown_lengths = [length]
    return drawdown_lengths
            

def compute_returns(equity):
    return (equity[1:] - equity[:-1])/equity[1:]


def time_in_market(returns):
    ticks_out = (abs(returns) < 1e-16)
    tim = 1 - sum(ticks_out)/len(returns)
    return tim


def avg_loss(returns):
    loss = returns[returns < 0]
    
    return loss.mean()


def avg_gain(returns):
    gain = returns[returns > 0]
    return gain.mean()


def avg_trade_win(trades):
    trade_wins = [trade.pnl_pct for trade in trades if trade.pnl>0]
    if len(trade_wins) > 0:
        return np.mean(trade_wins)
    else:
        return 0


def best_win(trades):
    trade_pnls = [trade.pnl_pct for trade in trades if trade.pnl_pct > 0]
    if len(trade_pnls) > 0:
        return max(trade_pnls)
    else:
        return 0


def worst_loss(trades):    
    trade_pnls = [trade.pnl_pct for trade in trades if trade.pnl_pct <= 0]
    if len(trade_pnls) > 0:
        return min(trade_pnls)
    else:
        return 0

def avg_trade_loss(trades):
    trade_loss = [trade.pnl_pct for trade in trades if trade.pnl<=0]
    if len(trade_loss) > 0:
        return np.mean(trade_loss)
    else:
        return 0


def wl_ratio(trades):
    wins = len([trade for trade in trades if trade.pnl>0])
    return wins/(len(trades) - wins)


def win_pct(trades):
    wins = len([trade for trade in trades if trade.pnl>0])
    return wins/len(trades)


def volatility(returns, periods=252, annualize=True):
    std = returns.std()
    if annualize:
        return std * np.sqrt(periods)
    return std


def sharpe(returns, periods=252, rf=0.0, annualize=True):
    res = returns.mean() / returns.std()
    if annualize: 
        return res * np.sqrt(periods)
    return res


def rolling_sharpe(returns, window = 50, periods=252, rf=0.0, annualize=True):
    
    d = returns.rolling(window).mean()
    q = returns.rolling(window).std()
    
    if annualize:
        return d/q * np.sqrt(periods)
    else:
        return d/q


def moving_average(prices, window):
    return sum(prices)/window


def dict_combinations(d):
    for vcomb in product(*d.values()):
        yield dict(zip(d.keys(), vcomb))
        