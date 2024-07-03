import pandas as pd
import numpy as np
import datetime
from itertools import product



# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
#Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

# Plot the results
#Daily_Drawdown.plot()
#Max_Daily_Drawdown.plot()
#pp.show()


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

def dict_combinations(d):
    for vcomb in product(*d.values()):
        yield dict(zip(d.keys(), vcomb))
        