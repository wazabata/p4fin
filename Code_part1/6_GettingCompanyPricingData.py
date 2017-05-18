## Starting clean for Part 5 - Automating getting list of s&p 500
import bs4 as bs
import pickle # Pickle is used to serialize any python object (save any object, e.g. variable) Here will save sp500 list to not hit wiki again and again
import requests
# imported in part 5
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web

def save_sp500_tickers():
    """
    Get the source code for wikipedia
    """
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml") # resp.txt is text of source code
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text # Shouldn't it be [1]
        tickers.append(ticker)

    with open("sp500tickers.pickle", 'wb') as f: # wb = write byte
        pickle.dump(tickers, f)

    return tickers



def get_data_from_yahoo(reload_sp500 = False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f) # Tickers list

    """
    When working with data online, grab once
    and then save locally to not crawl everytime
    """

    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)

    if not os.path.exists("stocks_dfs"):
        os.makedirs("stocks_dfs")

    for ticker in tickers:
        print(ticker)
        if not os.path.exists("stocks_dfs/{}.csv".format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
            except:
                df = web.DataReader(ticker.replace('.','-'), 'yahoo', start, end)
                # print("**** WARNING **** COULD NOT GET THE DATA FOR {}".ticker)
            df.to_csv("stocks_dfs/{}.csv".format(ticker))
        else:
            print("Alread have {}".format(ticker))


get_data_from_yahoo()
