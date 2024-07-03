from typing import Optional, List
from copy import deepcopy
from functools import partial
from itertools import product
import logging
import os

import pandas as pd
import numpy as np

from . import utils as _utils
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
        
        self.__results = results
        
        return self.__results
    
    
    def process_results(self, rolling: int = None):
        
        results = self.__results
        equity = pd.DataFrame(results['equity'])
        tick_dd, max_dd = _utils.compute_drawdown(equity)
        
        # returns
        results['final_equity'] = results['equity'][-1]
        
        
        returns = _utils.compute_returns(results['equity'])
        results['returns'] = returns
        results['log_returns'] = np.log(results['returns'])
        results['avg_loss'] = _utils.avg_loss(returns)
        
        # trades
        results['time_in_market'] = _utils.time_in_market(returns)
        
        # drawdown calculations
        results['tick_drawdown'] = tick_dd
        results['drawdown'] =  max_dd
        results['max_drawdown'] = min(tick_dd)
        results['avg_drawdown'] = tick_dd.mean()
        
        dd_length = _utils.compute_drawdown_length(tick_dd[0])
        results['avg_drawdown_length'] = np.mean(dd_length)
        results['longest_drawdown'] = max(dd_length)
        
        self.__results = results
        
        if rolling:
            self.add_rolling_drawdown(rolling)
            
        return self.__results
        
        
    def add_rolling_drawdown(self, window):
        
        equity = self.__results['equity']
        equity = pd.DataFrame(equity)
        dd, rolling_dd = _utils.compute_rolling_drawdown(equity, window)
        self.__results['rolling_dd'] = rolling_dd
    
    
    def plot(self):
        pass
    
    
    def show_results(self):
        pass
    
    
    def optimize(self, strategy, broker, param_grid,
                 target='final_equity', minimize=False):

        param_combinations = _utils.dict_combinations(param_grid)
        max_score = -np.inf
        sign = -1 if minimize else 1
                
        for params in param_combinations:
            strategy.params = params
            print(params)
            new_broker = deepcopy(broker)
            self.run(strategy, new_broker)
            self.process_results()
            
            new_score = self.__results[target] * sign
            if new_score > max_score:
                opt_results = self.__results
                max_score = opt_results[target]
                opt_results['best_params'] = params
        
        return opt_results
                
