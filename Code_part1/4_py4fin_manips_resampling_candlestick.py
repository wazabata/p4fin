import datetime as dt
import matplotlib.pyplot as plt # plt lets us utilize pyplot for charts etc
from matplotlib import style # Make graphs a little better
from matplotlib.finance import candlestick_ohlc # For part 4
import matplotlib.dates as mdates # For part 4
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

### Non-essential Part 2 Code
# pprint(df.head())
#
# # df.plot()
# # plt.show()
#
# # If want to plot something specific:
# df['Adj Close'].plot()
# plt.show()
# # print(df['Adj Close'])
# print(df[['Open','High']].head()) # Just print Open and high

## Part 3 - Manipulations
# Can map functions to a column easily to perform custom function
# Python functions a lot more efficient

## Make a new column - 100 MA (Moving Avg)

# My code for moving average:
# df['100ma'] = pd.rolling_mean(df['Adj Close'], window = 100)
# pprint(df.tail(5))

# # Sentdex code:
# For rolling(min_period, starts calc. with what it has (if at index = 2, takes avg of 0->2))
# df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean() # Commented out from part 3


'''
# Visualize this with matplotlib (without pandas)
# Each matplotlib object has a figure, figure contains subplots
# most of the tiem one subplot. If want to have mult graphes -> mult subplots
# Referred to as axix.
'''

## Resampling
'''
Want to count how many people walk in (want to know how many people come every hour)
With data of people coming every minute -> resample it to one hour (wut?)
With stock info, we have daily data -> resample to 10 day data (don't need millisecond data, can resample to minute or day data)
@DSNote: When resample, usually make a new df
'''
df_ohlc = df['Adj Close'].resample('10D').ohlc()  # resample to  10D = 10 days and get the mean
df_volume = df['Volume'].resample('10D').sum() # Get true volume, not the average vol

df_ohlc.reset_index(inplace = True) # Get the date
# df_volume.reset_index(inplace = True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) # Convert date to mdate

## Candlestick_ohlc wants from us ohlc + date (in dates format)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1) # 6 rows, 1 col, start at 0,0 top corner, how many rows span -> 5
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex = ax1) # Will have same x axis now
ax1.xaxis_date() # Take mdates and display as regular dates

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values,0) # First arg is x, 2nd is y, fill from 0 to y
plt.show() # Same thing with axis

## Next more complex manips and adding more data (all the S&P500)
