import pandas as pd
import numpy as np
import datetime



# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
#Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

# Plot the results
#Daily_Drawdown.plot()
#Max_Daily_Drawdown.plot()
#pp.show()


def compute_drawdown(equity):
    
    rolling_max = np.maximum.accumulate(equity)
    tick_dd = equity/rolling_max - 1.0
    max_dd = np.minimum.accumulate(tick_dd)
    
    return [tick_dd, max_dd]
            
        
        