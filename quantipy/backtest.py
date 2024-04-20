from functools import partial

import pandas as pd
import numpy as np

from quantipy.assets import Currency
from quantipy.trading import Broker, Strategy

class Backtester:
    
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Strategy,
                 initial_capital: float = 10_000,
                 currency: Currency = Currency('EUR'),
                 margin: float = 1.,
                 commission_fixed = 1.,
                 commission_pct = 0.002,
                 trade_on_close: bool = False,
                 hedging: bool = False,
                 exclusive_orders: bool = False
                 ):
        
        self.__data = data
        
        # Should check if this is ok
        # Assumes every entry has the same length
        self.__len_data = len(list(data.values())[0])
        
        # Partially initialize the broker object without data
        """ self.__broker = partial(
            Broker,
            initial_capital = initial_capital,
            currency = currency, 
            margin = margin,
            commission_fixed = commission_fixed,
            commission_pct = commission_pct,
            trade_on_close = trade_on_close,
            hedging = hedging,
            exclusive_orders = exclusive_orders
        ) """
        
        self.__broker = Broker(
            data = data,
            initial_capital = initial_capital,
            currency = currency, 
            margin = margin,
            commission_fixed = commission_fixed,
            commission_pct = commission_pct,
            trade_on_close = trade_on_close,
            hedging = hedging,
            exclusive_orders = exclusive_orders
        )
        
        self.__strategy = strategy

        self.__equity = None
        self.__results = None
        
    def run(self):
        start = self.__strategy.history + 1
        self.__equity = np.zeros(self.__len_data)
        
        # Running the backtest
        
        for i in range(start, self.__len_data):
            data = self.__data
            data = {k : v.iloc[:i] 
                    for k, v in data.items()}
                
            # Update the broker with new i
            broker = self.__broker._replace(__data = data, __i = i)
            
            # Process the orders
            broker._process_orders()
            
            # Update equity
            self.__equity[i] = broker.equity
            
            # Run strategy on new tick
            self.__strategy.next(broker)
        
        self.__equity = self.__equity[start:]
            
