from functools import partial

import pandas as pd
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
        self.__len_data = len(data.values()[0])
        
        # Partially initialize the broker object without data
        self.__broker = partial(
            Broker,
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
        self.__results = None
        
        def run(self):
            start = self.__strategy.history
            
            for i in range(start, len(self.__data)):
                
        