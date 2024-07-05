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
        
        self.__equity = None
        self.__backtest = None
        self.__results = None
        
        
    def run(self, strategy, broker, log_file='backtest.log', save_logs=False):

        self.__strategy = strategy
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
                self.__equity = self.__equity[:i+1]
                logger.warning('Out of equity.')
                break
            
            # Run strategy on new tick
            self.__strategy.next(broker)
        
        # Closing all remaining open trades
        for trade in broker.trades:
            trade.close()
        
        broker._process_orders()
        broker.logger.debug(broker.trades)
        
        # Final update to equity
        self.__equity[i] = broker.equity
        self.__equity = self.__equity[start:]
        
        results = {'equity': self.__equity,
                   'trades': broker.closed_trades,
                   'strategy': self.__strategy}
        
        self.__backtest = results
        
        return self.__backtest
    
    
    def process_results(self, results=None, rolling: int = 252):
        
        if results is None:
            results = self.__backtest
        
        equity = pd.DataFrame(results['equity'])
        tick_dd, max_dd = _utils.compute_drawdown(equity)
        
        # returns
        results['final_equity'] = results['equity'][-1]
        
        returns = _utils.compute_returns(results['equity'])
        results['returns'] = returns
        results['log_returns'] = np.log(results['returns'])
        results['cum_return'] = _utils.cum_return(results['equity'])
        
        results['avg_loss'] = _utils.avg_loss(returns)
        results['avg_gain'] = _utils.avg_gain(returns)
        results['volatility'] = _utils.volatility(returns)
        results['sharpe'] = _utils.sharpe(returns)
        
        returns = pd.DataFrame(returns)
        results['rolling_sharpe'] = _utils.rolling_sharpe(
            returns,
            window=rolling
        )
        
        # trades
        results['time_in_market'] = _utils.time_in_market(results['returns'])
        results['trade_count'] = len(results['trades'])
        results['best_trade_win'] = _utils.best_win(results['trades'])
        results['worst_trade_loss'] = _utils.worst_loss(results['trades'])
        results['avg_trade_win'] = _utils.avg_trade_win(results['trades'])
        results['avg_trade_loss'] = _utils.avg_trade_loss(results['trades'])
        results['wl_ratio'] = _utils.wl_ratio(results['trades'])
        results['win_pct'] = _utils.win_pct(results['trades'])
        
        # drawdown calculations
        results['tick_drawdown'] = tick_dd
        results['drawdown'] =  max_dd
        results['max_drawdown'] = min(tick_dd.values)[0]
        results['avg_drawdown'] = tick_dd.mean()[0]
        
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

    
    def show_results(self, results):
        title = 'Backtest Results'
        separator = '-'*40
        
        label = f"{'Metric':<15}{'Strategy':>10}{'Benchmark':>15}"
        
        # Strategy data
        cum_ret = f"{'Total Return:':<15}{results['cum_return']:>10.2%}"
        avg_loss = f"{'Avg loss (day):':<15}{results['avg_loss']:>10.2%}"
        avg_gain = f"{'Avg gain (day):':<15}{results['avg_gain']:>10.2%}"
        vol = f"{'Volatility:':<15}{results['volatility']:>10.4f}"
        sharpe = f"{'Sharpe Ratio:':<15}{results['sharpe']:>10.4f}"
        max_dd = f"{'Max Drawdown:':<15}{results['max_drawdown']:>10.2%}"
        avg_dd = f"{'Avg Drawdown:':<15}{results['avg_drawdown']:>10.2%}"
        dd_len = f"{'Avg DD Bars:':<15}{results['avg_drawdown_length']:>10.0f}"
        dd_len_max = f"{'Longest DD:':<15}{results['longest_drawdown']:>10}"
        
        # Trade data
        trades = f"{'Trades:':<15}{results['trade_count']:>10}"
        exposure = f"{'Time in Market:':<15}{results['time_in_market']:>10.2%}"
        best_win = f"{'Best Win:':<15}{results['best_trade_win']:>10.2%}"
        avg_trade_win = f"{'Avg Win:':<15}{results['avg_trade_win']:>10.2%}"
        worst_loss = f"{'Worst Loss:':<15}{results['worst_trade_loss']:>10.2%}"
        avg_trade_loss = f"{'Avg Loss:':<15}{results['avg_trade_loss']:>10.2%}"
        wl_ratio = f"{'Win/Loss ratio:':<15}{results['wl_ratio']:>10.2f}"
        win_pct = f"{'Win %:':<15}{results['win_pct']:>10.2%}"
        
        print(
            title,
            separator,
            label, separator,
            cum_ret, avg_gain, avg_loss, vol, sharpe,
            separator,
            max_dd, avg_dd, dd_len, dd_len_max,
            separator,
            trades, exposure,
            best_win, avg_trade_win, 
            worst_loss, avg_trade_loss,
            wl_ratio, win_pct,
            sep="\n"
        )
   
    
    def equity_plot(self, results):
        pass
    
    
    def underwater_plot(self, results):
        pass
    
    
    def optimize(self, strategy, broker, param_grid,
                 target='final_equity', minimize=False,
                 save_logs=False):

        param_combinations = _utils.dict_combinations(param_grid)
        max_score = -np.inf
        sign = -1 if minimize else 1
        broker.logger.debug('starting optimization...')
        
        for params in param_combinations:
            strategy.params = params
            print(params)
            broker.logger.debug(params)
            new_broker = deepcopy(broker)
            self.run(strategy, new_broker, save_logs=save_logs)
            self.process_results()
            
            new_score = self.__results[target] * sign
            print(f'Score: {new_score}')
            broker.logger.debug(f'Score: {new_score}')
            if new_score > max_score:
                opt_results = self.__results
                max_score = opt_results[target]
                opt_results['best_params'] = params
        
        opt_results['broker'] = broker
        return opt_results
                
