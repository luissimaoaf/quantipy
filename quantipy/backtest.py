from typing import Optional, List
from copy import deepcopy
from functools import partial
from itertools import product
import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)

from . import utils as _utils
from quantipy.assets import Currency
from quantipy.trading import Broker, Strategy


class Backtester:
    
    def __init__(self, data: dict[str:pd.DataFrame]):
        
        self.__data = data
        
        # Should check if this is ok
        # Assumes every entry has the same length
        self.__len_data = len(list(data.values())[0])
        
        # getting dates
        self.__dates = list(data.values())[0].index
        
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
        self.__benchmark = None
        
        
    def run(self, strategy, broker, log_file='backtest.log', save_logs=False):

        self.__strategy = strategy
        logger = broker.logger
        # not sure why +1 is bugging out
        start = self.__strategy.history + 2
        equity = np.zeros(self.__len_data)
        self.__equity = pd.Series(equity, index=self.__dates)
        self.__history = start
        
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
            self.__equity.iloc[i] = broker.equity
            
            if self.__equity.iloc[i] <= 0:
                self.__equity = self.__equity.iloc[:i+1]
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
        self.__equity.iloc[i] = broker.equity
        self.__equity = self.__equity.iloc[start:]
        
        results = {'equity': self.__equity,
                   'trades': broker.closed_trades,
                   'strategy': self.__strategy}
        
        self.__backtest = results
        
        return self.__backtest
    
    
    def process_results(
        self,
        results=None,
        rolling: int = 126,
        benchmark: str = None
    ):
        if benchmark:
            self.__benchmark = benchmark
        
        if results is None:
            results = self.__backtest
        
        bm_equity = self.__data[benchmark]['Close']

        tick_dd, max_dd = _utils.compute_drawdown(results['equity'])
        bm_tick_dd, bm_max_dd = _utils.compute_drawdown(bm_equity)
        
        # returns
        results['final_equity'] = results['equity'].iloc[-1]
        results['bm_final_equity'] = bm_equity[-1]
        
        returns = _utils.compute_returns(results['equity'])
        bm_returns = _utils.compute_returns(bm_equity)[self.__history:]
        
        results['returns'] = returns
        results['bm_returns'] = bm_returns
        
        results['log_returns'] = np.log(results['returns'])
        results['bm_log_returns'] = np.log(results['bm_returns'])
        
        results['cum_return'] = _utils.cum_return(results['equity'])
        results['bm_cum_return'] = _utils.cum_return(bm_equity)
        
        results['avg_loss'] = _utils.avg_loss(returns)
        results['avg_gain'] = _utils.avg_gain(returns)
        results['volatility'] = _utils.volatility(returns)
        results['sharpe'] = _utils.sharpe(returns)
        results['sortino'] = _utils.sortino(returns)
        
        results['bm_avg_loss'] = _utils.avg_loss(bm_returns)
        results['bm_avg_gain'] = _utils.avg_gain(bm_returns)
        results['bm_volatility'] = _utils.volatility(bm_returns)
        results['bm_sharpe'] = _utils.sharpe(bm_returns)
        results['bm_sortino'] = _utils.sortino(bm_returns)
        
        results['rolling_vol'] = _utils.rolling_volatility(
            returns,
            window=rolling
        )
        
        results['bm_rolling_vol'] = _utils.rolling_volatility(
            returns,
            window=rolling
        )
         
        results['rolling_sharpe'] = _utils.rolling_sharpe(
            returns,
            window=rolling
        )
        results['rolling_sortino'] = _utils.rolling_sortino(
            returns,
            window=rolling
        )
        
        greeks = _utils.greeks(returns, bm_returns)
        results['alpha'] = greeks['alpha']
        results['beta'] = greeks['beta']
        
        greeks_rs = _utils.rolling_greeks(returns, bm_returns, periods=126)
        greeks_ry = _utils.rolling_greeks(returns, bm_returns)
        results['rolling_alpha_6m'] = greeks_rs['alpha']
        results['rolling_beta_6m'] = greeks_rs['beta']
        results['rolling_alpha_1y'] = greeks_ry['alpha']
        results['rolling_beta_1y'] = greeks_ry['beta']
        
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
        results['max_drawdown'] = min(tick_dd.values)
        results['avg_drawdown'] = tick_dd.mean()
        # benchmark drawdown calculations
        results['bm_tick_drawdown'] = bm_tick_dd
        results['bm_drawdown'] =  bm_max_dd
        results['bm_max_drawdown'] = min(bm_tick_dd.values)
        results['bm_avg_drawdown'] = bm_tick_dd.mean()
        
        dd_length = _utils.compute_drawdown_length(tick_dd)
        results['avg_drawdown_length'] = np.mean(dd_length)
        results['longest_drawdown'] = max(dd_length)
        
        bm_dd_length = _utils.compute_drawdown_length(bm_tick_dd)
        results['bm_avg_drawdown_length'] = np.mean(bm_dd_length)
        results['bm_longest_drawdown'] = max(bm_dd_length)
        
        self.__results = results
        
        self.add_rolling_drawdown(rolling)
            
        return self.__results
        
        
    def add_rolling_drawdown(self, window):
        
        equity = self.__results['equity']
        equity = pd.DataFrame(equity)
        dd, rolling_dd = _utils.compute_rolling_drawdown(equity, window)
        self.__results['rolling_dd'] = rolling_dd

    
    def show_results(self):
        results = self.__results
        
        title = 'Backtest Results'
        separator = '-'*40
        
        label = f"{'Metric':<15}{'Strategy':>10}{'Benchmark':>15}"
        
        # Strategy data
        cum_ret = f"{'Total Return:':<15}{results['cum_return']:>10.2%}{results['bm_cum_return']:>15.2%}"
        avg_loss = f"{'Avg loss (day):':<15}{results['avg_loss']:>10.2%}{results['bm_avg_loss']:>15.2%}"
        avg_gain = f"{'Avg gain (day):':<15}{results['avg_gain']:>10.2%}{results['bm_avg_gain']:>15.2%}"
        vol = f"{'Volatility:':<15}{results['volatility']:>10.4f}{results['bm_volatility']:>15.4f}"
        sharpe = f"{'Sharpe Ratio:':<15}{results['sharpe']:>10.4f}{results['bm_sharpe']:>15.4f}"
        sortino = f"{'Sortino Ratio:':<15}{results['sortino']:>10.4f}{results['bm_sortino']:>15.4f}"
        max_dd = f"{'Max Drawdown:':<15}{results['max_drawdown']:>10.2%}{results['bm_max_drawdown']:>15.2%}"
        avg_dd = f"{'Avg Drawdown:':<15}{results['avg_drawdown']:>10.2%}{results['bm_avg_drawdown']:>15.2%}"
        dd_len = f"{'Avg DD Bars:':<15}{results['avg_drawdown_length']:>10.0f}{results['bm_avg_drawdown_length']:>15.0f}"
        dd_len_max = f"{'Longest DD:':<15}{results['longest_drawdown']:>10}{results['bm_longest_drawdown']:>15}"
        
        # Trade data
        trades = f"{'Trades:':<15}{results['trade_count']:>10}"
        exposure = f"{'Time in Market:':<15}{results['time_in_market']:>10.2%}{'100%':>15}"
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
            cum_ret, avg_gain, avg_loss, vol,
            sharpe, sortino,
            separator,
            max_dd, avg_dd, dd_len, dd_len_max,
            separator,
            exposure, trades, 
            best_win, avg_trade_win, 
            worst_loss, avg_trade_loss,
            wl_ratio, win_pct,
            sep="\n"
        )
   
    
    def equity_plot(self):
        plt.figure()
        plt.title('Equity Plot')
        
        strat_equity = self.__equity/self.__equity[0]-1
        history = self.__strategy.history
        plt.plot(strat_equity, label='Strategy')
        
        if self.__benchmark:
            benchmark = self.__data[self.__benchmark]['Close'].iloc[history:]
            plt.plot(benchmark/benchmark[0]-1, label='Benchmark')
        
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
        plt.legend()
        plt.axhline(0, color='black')
        plt.xticks(rotation=45)
        plt.show()
    
    
    def underwater_plot(self):
        plt.figure()
        plt.title('Underwater plot')
        
        plt.plot(self.__results['tick_drawdown'], label='Drawdown (tick)')
        plt.plot(self.__results['drawdown'], label='Maximum Drawdown')
        
        dd = self.__results['avg_drawdown']
        plt.axhline(dd, color='red', linestyle='--', label='Average Drawdown')
        
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
        plt.axhline(0, color='black')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
        
    
    def rolling_sharpe_plot(self):
        plt.figure()
        plt.title('Rolling Sharpe Ratio')
        plt.plot(self.__results['rolling_sharpe'])
        
        sharpe = self.__results['sharpe']
        plt.axhline(sharpe, color='red', linestyle='--')
        
        plt.axhline(0, color='black')
        plt.xticks(rotation=45)
        plt.show()


    def rolling_sortino_plot(self):
        plt.figure()
        plt.title('Rolling Sortino Ratio')
        plt.plot(self.__results['rolling_sortino'])
        
        sortino = self.__results['sortino']
        plt.axhline(sortino, color='red', linestyle='--')
        plt.axhline(0, color='black')
        plt.xticks(rotation=45)
        plt.show()
        
    
    def returns_plot(self):
        plt.figure()
        plt.title('Daily Returns')
        plt.plot(self.__results['returns'])
        plt.show()
        
        
    def rolling_beta_plot(self):
        plt.figure()
        plt.title('Rolling Beta to Benchmark')
        plt.plot(self.__results['rolling_beta_6m'], label='6-months')
        plt.plot(self.__results['rolling_beta_1y'], label='12-months')
        plt.axhline(
            self.__results['beta'], color='red', linestyle='--', label='Beta'
        )
        
        plt.axhline(0, color='black')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
    
    
    def rolling_vol_plot(self):
        plt.figure()
        plt.title('Rolling Volatility')
        plt.plot(self.__results['rolling_vol'], label='Strategy')
        plt.plot(self.__results['bm_rolling_vol'], label='Benchmark')
        plt.axhline(
            self.__results['volatility'], color='red', linestyle='--', label='Beta'
        )
        
        plt.axhline(0, color='black')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
    
    
    def show_report(self):
        self.show_results()
        self.equity_plot()
        self.underwater_plot()
        self.rolling_sharpe_plot()
        self.rolling_sortino_plot()
        self.rolling_beta_plot()
        self.returns_plot()
    
    
    def optimize(self, strategy, broker, param_grid,
                 target='final_equity', minimize=False,
                 save_logs=False, benchmark=None):

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
            self.process_results(benchmark=benchmark)
            
            new_score = self.__results[target] * sign
            print(f'Score: {new_score}')
            broker.logger.debug(f'Score: {new_score}')
            if new_score > max_score:
                opt_results = self.__results
                max_score = opt_results[target]
                opt_results['best_params'] = params
        
        opt_results['broker'] = broker
        
        self.__results = opt_results
        return opt_results
                
