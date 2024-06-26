
from typing import Optional, List
from math import copysign
import warnings
from copy import copy
import sys
import logging

import numpy as np
import pandas as pd

from .assets import Asset, Currency


class Order:
    def __init__(
        self,
        broker: 'Broker',
        asset: Asset,
        size: float,
        limit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        stop: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_id: int = None,
        parent_trade: Optional['Trade'] = None
    ):
        # Order info
        self.__parent_trade = parent_trade
        self.__order_id = order_id
        self.__broker = broker
        self.__asset = asset
        
        # Order size
        self.__size = size
        
        # Contingencies
        self.__limit = limit
        self.__stop_loss = stop_loss
        self.__stop = stop
        self.__take_profit = take_profit
        
    # Class methods
    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self
    
    def cancel(self):
        self.__brokers.order.remove(self)
        trade = self.__parent_trade
        
        if trade:
            if self is trade.stop_loss:
                trade._replace(sl_order=None)
            elif self is trade.take_profit:
                trade._replace(tp_order=None)
    
    # Property getters

    @property
    def asset(self) -> Asset:
        return self.__asset
    
    @property
    def size(self) -> float:
        '''
        Order size (negative if short order)
        '''
        return self.__size
    
    @property
    def parent_trade(self) -> Optional['Trade']:
        '''
        The order's parent trade (if it exists)
        '''
        return self.__parent_trade
    
    @property
    def order_id(self) -> int:
        return self.__order_id
    
    @property
    def limit(self) -> Optional[float]:
        return self.__limit
    
    @property
    def stop_loss(self) -> Optional[float]:
        return self.__stop_loss
    
    @property
    def stop(self) -> Optional[float]:
        return self.__stop
    
    @property
    def take_profit(self) -> Optional[float]:
        return self.__take_profit
    
    # Extra properties
    
    @property
    def is_long(self) -> bool:
        return self.size >= 0
    
    @property
    def is_short(self) -> bool:
        return not self.is_long


class Trade:
    def __init__(
        self,
        broker: 'Broker',
        asset: Asset,
        size: float,
        entry_bar: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        exit_bar: int = None,
        exit_price: int = None,
        trade_id: int = None
    ):
        self.__trade_id = trade_id
        self.__broker = broker
        self.__asset = asset
        
        # Trade size
        self.__size = size
        
        # Entry data
        self.__entry_bar = entry_bar
        self.__entry_price = entry_price
        
        # Contingencies
        self.__stop_loss = stop_loss
        self.__take_profit = take_profit
        self.__sl_order: Optional[Order] = None
        self.__tp_order: Optional[Order] = None
        
        # Closed trade
        self.__exit_bar = exit_bar
        self.__exit_price = exit_price
    
    # Class methods
    
    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self
    
    def _copy(self, **kwargs):
        return copy(self)._replace(**kwargs)
    
    def close(self, portion: float = 1.):
        """Place new `Order` to close `portion` of the trade at next market price."""
        assert 0 < portion <= 1, "portion must be a fraction between 0 and 1"
        size = copysign(max(1, round(abs(self.__size) * portion)), -self.__size)
        order = Order(self.__broker, self.asset, size, parent_trade=self)
        self.__broker.orders.insert(0, order)
    
    # Property getters
    
    @property
    def size(self) -> float:
        return self.__size
            
    @property
    def asset(self) -> Asset:
        return self.__asset
    
    @property
    def _sl_order(self) -> Optional[Order]:
        return self.__sl_order
    
    @property
    def _tp_order(self) -> Optional[Order]:
        return self.__tp_order
    
    @property
    def value(self) -> float:
        pass     
    
    # Extra properties
    
    @property
    def pnl(self):
        price = self.__exit_price or self.__broker.last_price(self.__asset)
        
        return self.__size * (price - self.__entry_price)
    
    @property
    def pnl_pct(self):
        price = self.__exit_price or self.__broker.last_price(self.__asset)
        
        return copysign(1, self.__size) * (price / self.__entry_price - 1)

    @property
    def value(self):
        """Trade total value in cash (volume x price)."""
        price = self.__exit_price or self.__broker.last_price(self.asset)
        
        return abs(self.__size) * price
    
    # SL/TP management API

    @property
    def stop_loss(self):
        """
        Stop-loss price at which to close the trade.

        This variable is writable. By assigning it a new price value,
        you create or modify the existing SL order.
        By assigning it `None`, you cancel it.
        """
        return self.__sl_order and self.__sl_order.stop

    @stop_loss.setter
    def stop_loss(self, price: float):
        self.__set_contingent('sl', price)

    @property
    def take_profit(self):
        """
        Take-profit price at which to close the trade.

        This property is writable. By assigning it a new price value,
        you create or modify the existing TP order.
        By assigning it `None`, you cancel it.
        """
        return self.__tp_order and self.__tp_order.limit

    @take_profit.setter
    def take_profit(self, price: float):
        self.__set_contingent('tp', price)

    def __set_contingent(self, type, price):
        assert type in ('sl', 'tp')
        assert price is None or 0 < price < np.inf
        attr = f'_{self.__class__.__qualname__}__{type}_order'
        order: Order = getattr(self, attr)
        if order:
            order.cancel()
        if price:
            kwargs = {'stop': price} if type == 'sl' else {'limit': price}
            order = self.__broker._new_order(self.asset, -self.size, parent_trade=self, **kwargs)
            setattr(self, attr, order)

  
class Position:
    def __init__(
        self,
        asset: Asset,
        size: float
    ):
        self.__asset = asset
        self.size = size
        
    # Main methods
    
    # Property getters
    
    @property
    def asset(self) -> Asset:
        return self.__asset


class Broker:
    def __init__(
        self,
        data: dict[str: pd.DataFrame] = None,
        initial_capital: float = 10_000,
        currency: Currency = Currency('EUR'),
        commission_pct: float = 0.002,
        commission_fixed: float = 1.0,
        margin: float = 1.0,
        trade_on_close: bool = False,
        hedging: bool = False,
        exclusive_orders: bool = False
    ):
        if not currency.is_cash:
            raise TypeError('ERROR: You must provide an initial cash position.')
        
        # Logging
        self.logger = logging.getLogger('broker_log')
        
        # A market is a dictionary with symbols : assets
        self.__data = data
        self.__i = len(list(data.values())[0]) if data else None
        
        # Cash account and costs
        self.__initial_capital = initial_capital
        self.__base_currency = currency
        self.cash = Position(currency, initial_capital)
        self.__commission_pct = commission_pct
        self.__commission_fixed = commission_fixed
        
        # Margin and leverage
        self.__margin = margin
        self.__leverage = 1 / self.__margin
        
        # Execution specification
        self.__trade_on_close = trade_on_close
        self.__hedging = hedging
        self.__exclusive_orders = exclusive_orders
        
        # Trade and order data
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.positions: List[Position] = []
        self.closed_trades: List[Trade] = []
        self.trade_no = 0
    
    # Property getters
    
    @property
    def _i(self):
        return self.__i
    
    @property
    def equity(self):
        return self.cash.size + sum(trade.pnl for trade in self.trades)
    
    @property
    def margin_available(self):
        margin_used = sum(trade.value/self.__leverage 
                          for trade in self.trades)
        return max(0, self.equity - margin_used)
    
    # Main methods
    
    def last_price(self, asset) -> float:
        data = self.__data[asset.symbol]
        return data['Close'].iloc[-1]
    
    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self
    
    def _adjusted_price(self, asset, size=None, price=None) -> float:
        """
        Long/short `price`, adjusted for commisions.
        In long positions, the adjusted price is a fraction higher, while in short positions it is a fraction lower.
        """
        price = price or self.last_price(asset)
        factor = 1 + copysign(self.__commission_pct, size)
        pct_adjust = price * factor
        fixed_adjust = price + copysign(self.__commission_fixed, size)
        
        if size > 0:
            return max(pct_adjust, fixed_adjust)
        else:
            return min(pct_adjust, fixed_adjust)
      
    def _new_order(
        self,
        asset: Asset,
        size: float,
        limit: Optional[float] = None,
        stop: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        parent_trade: Optional[Trade] = None,
        order_id: int = None
    ):
        is_long = (size > 0)
        adjusted_price = self._adjusted_price(asset, size=size)
        
        # Common sense check
        if is_long:
            cond1 = (stop_loss or -np.inf) < (limit or stop or adjusted_price)
            cond2 = (limit or stop or adjusted_price) < (take_profit or np.inf)
            if not (cond1 and cond2):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({stop_loss}) < LIMIT ({limit or stop or adjusted_price}) < TP ({take_profit})")
        else:
            cond1 = (take_profit or -np.inf) < (limit or stop or adjusted_price)
            cond2 = (limit or stop or adjusted_price) < (stop_loss or np.inf)
            if not cond1 and cond2:
                raise ValueError(
                    "Short orders require: "
                    f"TP ({take_profit}) < LIMIT ({limit or stop or adjusted_price}) < SL ({stop_loss})")
        
        # Create the order        
        order = Order(self, asset, size, limit, stop_loss, stop, take_profit, 
                      order_id, parent_trade)
        
        if parent_trade:
            # Add the order to the order queue
            self.orders.insert(0, order)
        else:
            if self.__exclusive_orders:
                for old_order in self.orders:
                    if not old_order.is_contingent:
                        old_order.cancel()
                
                for old_trade in self.trades:
                    old_trade.close()
            
            self.orders.append(order)
        
        return order
    
    def _reduce_trade(
        self,
        trade: Trade, 
        price: float, 
        size: float, 
        time_index: int
    ):
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            # Reduce existing trade ...
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            # ... by closing a reduced copy of it
            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            self.trades.append(close_trade)

        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade: Trade, price: float, time_index: int):
        self.logger.debug('Closed trade')
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        self.closed_trades.append(trade._replace(exit_price=price, exit_bar=time_index))
        self.cash.size += trade.pnl

    def _open_trade(self, asset: Asset, price: float, size: int,
                    stop_loss: Optional[float], take_profit: Optional[float], time_index: int):
        trade = Trade(self, asset, size, time_index, price, 
                      trade_id=self.trade_no)

        self.trade_no += 1
        
        self.trades.append(trade)
        # Create SL/TP (bracket) orders.
        # Make sure SL order is created first so it gets adversarially processed before TP order
        # in case of an ambiguous tie (both hit within a single bar).
        # Note, sl/tp orders are inserted at the front of the list, thus order reversed.
        if take_profit:
            trade.take_profit = take_profit
        if stop_loss:
            trade.stop_loss = stop_loss
        
    def _process_orders(self):
        
        reprocess_orders = False
        
        for order in list(self.orders):
            
            asset = order.asset
            data = self.__data[asset.symbol]
            
            open = data['Open'].iloc[-1]
            high = data['High'].iloc[-1]
            low = data['Low'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            
            if order not in self.orders:
                continue
            
            stop_price = order.stop
            if stop_price:
                is_stop_hit = (
                    (high > stop_price) if order.is_long else
                    (low < stop_price)
                )
                
                if is_stop_hit:
                    self.logger.debug('stop hit')
                # If stop price wasn't hit, the order isn't active yet
                if not is_stop_hit:
                    continue
                    
                order._replace(stop = None)
    
            if order.limit:
                is_limit_hit = (
                    low < order.limit if order.is_long else
                    high > order.limit
                )
                
                if is_limit_hit:
                    self.logger.debug('limit hit')
                
                is_limit_hit_before_stop = (
                    is_limit_hit and
                    (order.limit < (stop_price or -np.inf)
                    if order.is_long else
                    order.limit > (stop_price or np.inf))
                )

                # Ignore order if limit wasn't hit yet or the limit was hit before the stop was triggered
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue
                
                price = (
                    min(stop_price or open, order.limit)
                    if order.is_long else
                    max(stop_price or open, order.limit)
                )
                
            else:
                # Market order
                price = prev_close if self.__trade_on_close else open
                price = (
                    max(price, stop_price or -np.inf)
                    if order.is_long else
                    min(price, stop_price or np.inf)
                )
            
            is_market_order = not order.limit and not stop_price
            time_index = (
                self._i - 1
                if is_market_order and self.__trade_on_close else
                self._i
            )
            
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                
                size = copysign(min(abs(_prev_size), abs(order.size)), 
                                order.size)
                
                # trade is still open
                if trade in self.trades:
                    
                    self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades
                if order in (trade._sl_order,
                             trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders
                else:
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                    
                continue
            
            size = order.size
            adjusted_price = self._adjusted_price(asset, size, price)
            
            if -1 < size < 1:
            
                size = copysign(int((self.margin_available * self.__leverage * abs(size)) // adjusted_price), size)
                # Not enough cash/margin even for a single unit
                if not size:
                    self.orders.remove(order)
                    continue
            assert size == round(size)
            need_size = int(size)
            
            if not self.__hedging:
                # Fill position by FIFO closing/reducing existing opposite-facing trades.
                # Existing trades are closed at unadjusted price, because the adjustment
                # was already made when buying.
                for trade in list(self.trades):
                    if trade.is_long == order.is_long:
                        continue
                    assert trade.size * order.size < 0

                    # Order size greater than this opposite-directed existing trade,
                    # so it will be closed completely
                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)
                        need_size += trade.size
                    else:
                        # The existing trade is larger than the new order,
                        # so it will only be closed partially
                        self._reduce_trade(trade, price, need_size, time_index)
                        need_size = 0

                    if not need_size:
                        break

            # If we don't have enough liquidity to cover for the order, cancel it
            if abs(need_size) * adjusted_price > self.margin_available * self.__leverage:
                self.orders.remove(order)
                continue

            # Open a new trade
            if need_size:
                self._open_trade(asset,
                                 adjusted_price,
                                 need_size,
                                 order.stop_loss,
                                 order.take_profit,
                                 time_index)

                # We need to reprocess the SL/TP orders newly added to the queue.
                # This allows e.g. SL hitting in the same bar the order was open.
                # See https://github.com/kernc/backtesting.py/issues/119
                if order.stop_loss or order.take_profit:
                    if is_market_order:
                        reprocess_orders = True
                    elif (low <= (order.stop_loss or -np.inf) <= high or
                          low <= (order.take_profit or -np.inf) <= high):
                        warnings.warn(
                            f"({data.index[-1]}) A contingent SL/TP order would execute in the "
                            "same bar its parent stop/limit order was turned into a trade. "
                            "Since we can't assert the precise intra-candle "
                            "price movement, the affected SL/TP order will instead be executed on "
                            "the next (matching) price/bar, making the result (of this trade) "
                            "somewhat dubious. "
                            "See https://github.com/kernc/backtesting.py/issues/119",
                            UserWarning)

            # Order processed
            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()
    
    
class Strategy:
    
    def __init__(self,
                 broker: Broker,
                 assets: List[Asset],
                 params: dict = {}):
        self.__broker = broker
        self.__assets = assets
        self.__params = params
    
    @property
    def history(self) -> int:
        try:
            return self.params['history']
        except:
            return 0
    
    @property
    def broker(self) -> Broker:
        return self.__broker
    
    @property
    def assets(self) -> List[Asset]:
        return self.__assets
    
    @property
    def params(self) -> dict:
        return self.__params
    
    @params.setter
    def params(self, params: dict = {}):
        self.__params = params
    
    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self): return '.9999'
        
    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)
    
    def buy(self, 
            asset: Asset,
            size: float = _FULL_EQUITY,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None
            ):
        assert 0 < size < 1 or round(size) == size, \
            'Size must be a positive fraction of equity or a positive whole number of units'
        
        return self.__broker._new_order(asset, size, 
                                        limit, stop, stop_loss, take_profit)
        
    def sell(self, 
            asset: Asset,
            size: float = _FULL_EQUITY,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None
            ):
        assert 0 < size < 1 or round(size) == size, \
            'Size must be a positive fraction of equity or a positive whole number of units'
        
        return self.__broker._new_order(asset, -size, 
                                        limit, stop, stop_loss, take_profit)
    
    def next(self):
        pass
    
    
if __name__ == '__main__':
    print("It worked!")