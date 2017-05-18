import datetime as dt
import matplotlib.pyplot as plt # plt lets us utilize pyplot for charts etc
from matplotlib import style # Make graphs a little better
import pandas as pd
import pandas_datareader.data as web # Replaces pandas.data.io (or pandas.io.data) - Use to grab data from Yahoo Finance page
from pprint import pprint

style.use('ggplot') # Specify style to use

start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)

df = web.DataReader('TSLA', 'yahoo', start, end) # Tesla Ticker
pprint(df.tail(6)) # Return last 6 columns

"""
@StockNote:
Adj Close -> Adjusted for stock splits, e.g. when company decides
that price is too high, every share now 2 shares. One share at 1000$ -> now
have 2 stocks at 500$. You'll therefore see a drop in the price
Splits are done because if price too high, higher barrier to entry when it comes
to buying stocks
"""
