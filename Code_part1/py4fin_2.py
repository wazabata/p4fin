import datetime as dt
import matplotlib.pyplot as plt # plt lets us utilize pyplot for charts etc
from matplotlib import style # Make graphs a little better
import pandas as pd
import pandas_datareader.data as web # Replaces pandas.data.io (or pandas.io.data) - Use to grab data from Yahoo Finance page
from pprint import pprint

style.use('ggplot') # Specify style to use

# start = dt.datetime(2000,1,1)
# end = dt.datetime(2016,12,31)
#
# df = web.DataReader('TSLA', 'yahoo', start, end) # Tesla Ticker

"""
@StockNote:
Adj Close -> Adjusted for stock splits, e.g. when company decides
that price is too high, every share now 2 shares. One share at 1000$ -> now
have 2 stocks at 500$. You'll therefore see a drop in the price
Splits are done because if price too high, higher barrier to entry when it comes
to buying stocks
"""

# df.to_csv('tsla.csv') # By default, the index is the Date, which is desirable
# But when reading the csv file, it's not going to know that Date column is an
# index

## Read the csv that was saved above
df = pd.read_csv('tsla.csv')
# pprint(df.head()) # Date is now a column, want a date time index

## Add the necessary arguments to read_csv
df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 0) # parse_dates: if give it a bool, tries to parse the index col
pprint(df.head())

# df.plot()
# plt.show()

# If want to plot something specific:
df['Adj Close'].plot()
plt.show()
# print(df['Adj Close'])
print(df[['Open','High']].head()) # Just print Open and high
