from quantipy.assets import Asset, Currency
from typing import Optional, List


class Order:
    def __init__(
        self,
        order_id: int,
        broker: 'Broker',
        size: float,
        parent_trade: Optional['Trade'] = None,
        limit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        stop: Optional[float] = None,
        take_profit: Optional[float] = None,
        is_contingent: bool = False
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
        self.__is_contingent = is_contingent
        
    # Class methods
    
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
        trade_id: int,
        size: float,
        value: float,
        entry_bar: int,
        entry_price: float,
        entry_time,
        pnl: float = 0,
        pnl_pct: float = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        exit_bar: int = None,
        exit_price: int = None,
        exit_time = None
    ):
        self.__trade_id = trade_id
        
        # Trade size
        self.__size = size
        self.__value = value
        
        # Entry data
        self.__entry_bar = entry_bar
        self.__entry_price = entry_price
        self.__entry_time = entry_time
        
        # Current PnL
        self.__pnl = pnl
        self.__pnl_pct = pnl_pct
        
        # Contingencies
        self.__stop_loss = stop_loss
        self.__take_profit = take_profit
        
        # Closed trade
        self.__exit_bar = exit_bar
        self.__exit_price = exit_price
        self.__exit_time = exit_time
        
        
class Position:
    def __init__(
        self,
        asset: Asset,
        size: float,
        pnl: float = 0,
        pnl_pct: float = 0
    ):
        self.__asset = asset
        self.__size = size
        
        self.__pnl = pnl
        self.__pnl_pct = pnl_pct
    
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
        data,
        cash: Position,
        commision: float = 0.002,
        margin: float = 1.0,
        trade_on_close: bool = False
    ):
        if not cash.asset.is_cash:
            raise TypeError('ERROR: You must provide an initial cash position.')
        
        # Historical data up to present
        self.__data = data
        self.__idx = len(self.__data) - 1
        
        # Cash account and costs
        self.__cash = cash
        self.__commission = commision
        
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
    
    def _process_orders(self):
        pass
    
    def _new_order(self):
        pass