import yfinance as yf

import quantipy.assets
import quantipy.trading


def currency_test():
    print('Starting currency test...')
    # Creating a currency object
    usd = quantipy.assets.Currency('USD')
    print('Currency: ', usd.currency)
    print('Is cash?: ', usd.is_cash)


def equity_test():
    print('Starting equity test...')
    # Creating an Equity object
    aapl = quantipy.assets.Equity(name='Apple', symbol='AAPL')
    print('Stock name: ', aapl.name)
    print('Symbol: ', aapl.symbol)
    print('Is cash?: ', aapl.is_cash)


def position_test():
    print('Starting positions test...')
    # Creating positions
    eur = quantipy.assets.Currency('EUR')
    cash_position = quantipy.trading.Position(eur, 10_000)
    boeing = quantipy.assets.Equity('Boeing', 'BA')
    boeing_position = quantipy.trading.Position(boeing, 10_000)

    
if __name__ == '__main__':
    print('Starting unit tests...')
    
    currency_test()
    equity_test()
    position_test()
    
    print('Done!')