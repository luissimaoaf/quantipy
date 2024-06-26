from typing import Optional, List
from copy import deepcopy
from functools import partial
from itertools import product
import logging
import os

import pandas as pd
import numpy as np

import quantipy.utils
from quantipy.assets import Currency
from quantipy.trading import Broker, Strategy


class Backtester:
    
    def __init__(self, data: dict[str:pd.DataFrame]):
        
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
        
        self.__broker = None
        self.__strategy = None
        self.__equity = None
        self.__results = None
        
        
    def run(self, strategy, broker, log_file='backtest.log', save_logs=False):

        self.__strategy = strategy
        self.__broker = broker
        logger = broker.logger
        # not sure why +1 is bugging out
        start = self.__strategy.history + 2
        self.__equity = np.zeros(self.__len_data)
        
        if save_logs:
            logger.setLevel(logging.DEBUG)
            # create file handler which logs even debug messages
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        
        # Running the backtest
        logger.debug('Starting backtest...')
        
        for i in range(start, self.__len_data):
            data = self.__data
            data = {k : v.iloc[:i] for k, v in data.items()}
                
            # Update the broker with new i
            broker = broker._replace(data = data, i = i)
            
            # Process the orders
            broker._process_orders()
            
            # Update equity
            self.__equity[i] = broker.equity
            
            if self.__equity[i] <= 0:
                self.__equity[i] = 0
                logger.warning('Out of equity.')
                break
            
            # Run strategy on new tick
            self.__strategy.next(broker)
        
        # Closing all remaining open trades
        for trade in self.__broker.trades:
            trade.close()
        
        broker._process_orders()
        
        # Final update to equity
        self.__equity[i] = broker.equity
        self.__equity = self.__equity[start:]
        
        results = {'equity': self.__equity,
                   'trades': broker.closed_trades,
                   'data': data,
                   'strategy': self.__strategy}
        
        self.process_results(results)
        
        return self.__results
    
    
    def process_results(self, results):
        tick_dd, max_dd = quantipy.utils.compute_drawdown(results['equity'])
        
        #placeholder
        results['final_equity'] = results['equity'][-1]
        results['tick_drawdown'] = tick_dd
        results['drawdown'] =  max_dd
        results['max_drawdown'] = min(tick_dd)
        
        self.__results = results
    
    
    def plot(self):
        pass
    
    
    def show_results(self):
        pass
    
    
    def optimize(self, strategy, broker, param_grid, target='final_equity'):

        def dict_combinations(d):
            for vcomb in product(*d.values()):
                yield dict(zip(d.keys(), vcomb))

        param_combinations = dict_combinations(param_grid)
        max_score = -np.inf
        best_params = {}
                
        for params in param_combinations:
            strategy.params = params
            print(params)
            new_broker = deepcopy(broker)
            self.run(strategy, new_broker)
            
            if self.__results[target] > max_score:
                opt_results = self.__results
                max_score = opt_results[target]
                opt_results['best_params'] = params
        
        return opt_results
                
