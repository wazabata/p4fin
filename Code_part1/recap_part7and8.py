import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


### First get the tickers from Wikipedia
def save_sp500_tickers():
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    # Save with pickle
    if os.path.exists("sp500tickers.pickle"):
        print("SP500 Pickle file already exists")
    else:
        with open("sp500tickers.pickle", "wb") as f:
            pickle.dump(f)


def get_data_from_yahoo(reload_sp500 = False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)

    if not os.path.exists("stocks_dfs"):
        os.nmakedirs("stocks_dfs")

    # Have list of sp500 tickers, now get the data from yahoo
    for ticker in tickers:
        if os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            print("Data file already exists")
        else:
            try:
                stock = web.DataReader(ticker, 'yahoo', start, end)
            except:
                stock = web.DataReader(ticker.replace('.','-'), 'yahoo', start, end)
            df.to_csv("stocks_dfs/{}.csv".format(ticker))

## Part 7
# Compiling the data to one big df

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv("stocks_dfs/{}.csv".format(ticker))
        df.set_index('Date', inplace = True)

        df.rename(columns = {'Adj Close': ticker}, inplace = True)
        df.drop(['Open','High', 'Low', 'Close', 'Volume'], axis = 1, inplace = True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how = 'outer')

        if count % 10 == 0:
            print("{} out of {}".format(count, len(tickers)))

    main_df.to_csv("sp500_joined_closes.csv")


def visualize_data():
    df = pd.read_csv("sp500_joined_closes.csv")
    df_corr = df.corr()

    data = df_corr.values

    fig = plt.figure()
    ax = fig.add_subplot(111)

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
    fig.colorbar = heatmap
    ax.set_xticks(np.arange(data.shape[0]) + .5, minor = False)
    ax.set_yticks(np.arange(data.shape[1]) + .5, minor = False)
    ax.invert_yaxis
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation = 90)
    heatmap.set_clim(-1,1) # limit of these colors, -1 is the min and 1 is the max
    # # For covariance, don't bound the scale!
    plt.tight_layout()
    plt.show()

visualize_data()
