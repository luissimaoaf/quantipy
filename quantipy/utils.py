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
            
            
def dict_combinations(d):
    for vcomb in product(*d.values()):
        yield dict(zip(d.keys(), vcomb))
        