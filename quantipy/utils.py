import pandas as pd
import numpy as np
import datetime
from itertools import product


# tools
def dict_combinations(d):
    for vcomb in product(*d.values()):
        yield dict(zip(d.keys(), vcomb))


# stats
def cum_return(equity):
    return equity[-1]/equity[0] - 1


def compute_drawdown(equity):
    
    rolling_max = equity.cummax()
    tick_dd = equity/rolling_max - 1.0
    max_dd = tick_dd.cummin()
    
    return [tick_dd, max_dd]


def compute_rolling_drawdown(equity, window):
    
    cum_max = equity.cummax()
    rolling_max = equity.rolling(window, min_periods=1).max()
    tick_dd = equity/cum_max - 1.0
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
    return (equity - equity.shift(1))/equity.shift(1)


def match_vol(returns, benchmark):
    vol = volatility(returns)
    bm_vol = volatility(benchmark)
    scaled_ret = returns * bm_vol/vol
    match_equity = (1+scaled_ret).cumprod().fillna(1)
    return match_equity


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


def avg_trade_duration(trades):
    dur = [trade.exit_bar - trade.entry_bar for trade in trades]
    return np.mean(dur)


def avg_win_duration(trades):
    dur = [trade.exit_bar - trade.entry_bar for trade in trades
           if trade.pnl > 0]
    return np.mean(dur)


def avg_loss_duration(trades):
    dur = [trade.exit_bar - trade.entry_bar for trade in trades
           if trade.pnl <= 0]
    
    return np.mean(dur)


def wl_ratio(trades):
    wins = len([trade for trade in trades if trade.pnl>0])
    if len(trades) == wins:
        return np.inf
    else:
        return wins/(len(trades) - wins)


def win_pct(trades):
    wins = len([trade for trade in trades if trade.pnl>0])
    return wins/len(trades)


def volatility(returns, periods=252, annualize=True):
    std = returns.std()
    if annualize:
        return std * np.sqrt(periods)
    return std


def rolling_volatility(returns, window=126, periods=252, annualize=True):
    std = returns.rolling(window).std()
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
    res = d/q
    
    if annualize:
        return res * np.sqrt(periods)
    else:
        return res
    
    
def sortino(returns, periods=252, rf=0.0, annualize=True): 
    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))
    res = returns.mean() / downside
    
    if annualize:
        return res * np.sqrt(periods)
    else:
        return res

def rolling_sortino(returns, window = 50, periods=252, rf=0.0, annualize=True):
    downside = (
        returns.rolling(window).apply(
            lambda x: (x.values[x.values < 0] ** 2).sum()
        )
        / window
    )
    res = (returns.rolling(window).mean() / np.sqrt(downside))
    
    if annualize:
        return res * np.sqrt(periods)
    else:
        return res
    

def moving_average(prices, window):
    return sum(prices)/window
        
        
def greeks(returns, benchmark, periods=252.0):
    """Calculates alpha and beta of the portfolio"""

    # find covariance
    matrix = np.cov(returns.fillna(0), benchmark.fillna(0))
    beta = matrix[0, 1] / matrix[1, 1]

    # calculates measures now
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods

    return pd.Series(
        {
            "beta": beta,
            "alpha": alpha,
        }
    ).fillna(0)


def rolling_greeks(returns, benchmark, periods=252):
    """Calculates rolling alpha and beta of the portfolio"""
    df = pd.DataFrame(
        data={
            "returns": returns,
            "benchmark": benchmark,
        }
    )
    df = df.fillna(0)
    corr = df.rolling(int(periods)).corr().unstack()["returns"]["benchmark"]
    std = df.rolling(int(periods)).std()
    beta = corr * std["returns"] / std["benchmark"]

    alpha = df["returns"].mean() - beta * df["benchmark"].mean()

    alpha = alpha * periods
    return pd.DataFrame(index=returns.index, data={"beta": beta, "alpha": alpha})