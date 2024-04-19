
from typing import Optional, List
from math import copysign

import numpy as np
import pandas as pd

from quantipy.assets import Asset, Currency



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
    
    # Property getters
    
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
        value: float,
        entry_bar: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        exit_bar: int = None,
        exit_price: int = None,
        exit_time = None,
        trade_id: int = None
    ):
        self.__trade_id = trade_id
        self.__broker = broker
        self.__asset = asset
        
        # Trade size
        self.__size = size
        self.__value = value
        
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
    
    # Property getters
    
    @property
    def size(self) -> float:
        return self.__size
            
    # Extra properties
    
    @property
    def pnl(self):
        price = self.__exit_price or self.__broker.last_price
        
        return self.__size * (price - self.__entry_price)
    
    @property
    def pnl_pct(self):
        price = self.__exit_price or self.__broker.last_price
        
        return copysign(1, self.__size) * (price / self.__entry_price - 1)

    @property
    def value(self):
        """Trade total value in cash (volume x price)."""
        price = self.__exit_price or self.__broker.last_price
        
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
            order = self.__broker.new_order(-self.size, trade=self, **kwargs)
            setattr(self, attr, order)
        
class Position:
    def __init__(
        self,
        asset: Asset,
        size: float
    ):
        self.__asset = asset
        self.__size = size
        
    # Main methods
    
    # Property getters
    
    @property
    def asset(self) -> Asset:
        return self.__asset
    
    @property
    def size(self) -> float:
        return self.__size
       
        
class Broker:
    def __init__(
        self,
        data: dict[str: pd.DataFrame],
        initial_capital: float = 10_000,
        currency: Currency = Currency('EUR'),
        commission_pct: float = 0.002,
        commission_fixed: float = 1.0,
        margin: float = 1.0,
        trade_on_close: bool = False
    ):
        if not currency.is_cash:
            raise TypeError('ERROR: You must provide an initial cash position.')
        
        # A market is a dictionary with symbols : assets
        self.__data = data
        
        # Cash account and costs
        self.__initial_capital = initial_capital
        self.__base_currency = currency
        self.cash = Position(currency, initial_capital)
        self.__commission_pct = commission_pct
        self.__commission_pct = commission_fixed
        
        # Margin and leverage
        self.__margin = margin
        self.__leverage = 1 / self.__margin
        
        # Execution specification
        self.__trade_on_close = trade_on_close
        
        # Trade and order data
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.positions: List[Position] = []
        self.closed_trades: List[Trade] = []
        
    def _adjusted_price(self, size=None, price=None) -> float:
        """
        Long/short `price`, adjusted for commisions.
        In long positions, the adjusted price is a fraction higher, while in short positions it is a fraction lower.
        """
        price = price or self.last_price
        factor = 1 + copysign(self._commission_pct, size)
        pct_adjust = price * factor
        fixed_adjust = price + copysign(self.__commission_fixed, size)
        
        if size > 0:
            return max(pct_adjust, fixed_adjust)
        else:
            return min(pct_adjust, fixed_adjust)
    
    def _new_order(
        self,
        order_id: int,
        asset: Asset,
        size: float,
        limit: Optional[float] = None,
        stop: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        parent_trade: Optional[Trade] = None
    ):
        is_long = (size > 0)
        adjusted_price = self._adjusted_price(size)
        
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
        
        # Add the order to the order queue
        self.orders.insert(0, order)
        
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
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        self.closed_trades.append(trade._replace(exit_price=price, exit_bar=time_index))
        self._cash += trade.pnl
        
    def _process_orders(self):
        
        for order in list(self.orders):
            
            asset = order.asset.symbol
            data = self.data[asset]
            
            open = data['Open'][-1]
            high = data['High'][-1]
            low = data['Low'][-1]
            prev_close = data['Close'][-2]
            
            if order not in self.orders:
                continue
            
            stop_price = order.stop
            if stop_price:
                is_stop_hit = (
                    (high > stop_price) if order.is_long else
                    (low < stop_price)
                )
                # If stop price wasn't hit, the order isn't active yet
                if not is_stop_hit:
                    continue
                    
                order._replace(stop = None)
    
            if order.limit:
                is_limit_hit = (
                    low < order.limit if order.is_long else
                    high > order.limit
                )
                
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
            
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                
                size = copysign(min(abs(_prev_size), abs(order.size)), 
                                order.size)
                
                # trade is still open
                if trade in self.trades:
                    self._reduce_trade(trade, price, size)