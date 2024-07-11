from abc import ABC
from typing import Optional
from datetime import datetime


class Asset:
    def __init__(self):
        pass


class Currency(Asset):
    def __init__(
        self,
        currency: str = 'EUR'
    ):
        super().__init__()
        self.__currency = currency
        
        self.__cash_like = True
    
    # Property getters
    
    @property
    def currency(self) -> str:
        return self.__currency
    
    @property
    def is_cash(self) -> bool:
        return self.__cash_like
    
    
class Equity(Asset):
    def __init__(
        self,
        symbol: str
    ):
        super().__init__()
        self.__type = 'equity'
        self.__cash_like = False
        self.__symbol = symbol
        self.__data = None
        
    def set_data(self, data):
        self.__data = data
        
    # Property getters
    
    @property
    def type(self) -> str:
        return self.__type
    
    @property
    def is_cash(self) -> bool:
        return self.__cash_like
    
    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def symbol(self) -> str:
        return self.__symbol
    
    @property
    def data(self):
        return self.__data
    
    
class Option(Asset):
    def __init__(
        self,
        symbol: str,
        underlying: Asset,
        strike: float,
        expiration: datetime,
        type: str = 'call',
        style: str = 'european'
    ):
        super().__init__()
        
        self.symbol = symbol
        self.__underlying = underlying
        self.__strike = strike
        self.__expiration = expiration
        self.__type = type
        self.__style = style
        
    @property
    def underlying(self):
        return self.__underlying
    
    @property
    def strike(self):
        return self.__strike
    
    @property
    def expiration(self):
        return self.__expiration
    
    @property
    def type(self):
        return self.__type
    
    @property 
    def sign(self):
        return 1 if (self.__type == 'call') else -1
    
    @property
    def style(self):
        return self.__style


    def time_to_maturity(self, t: datetime):
        return (self.expiration - t).days