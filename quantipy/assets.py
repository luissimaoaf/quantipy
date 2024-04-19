from abc import ABC
from typing import Optional


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
        name: str,
        symbol: str
    ):
        super().__init__()
        self.__cash_like = False
        self.__name = name
        self.__symbol = symbol
        self.__data = None
        
    def set_data(self, data):
        self.__data = data
        
    # Property getters
    
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