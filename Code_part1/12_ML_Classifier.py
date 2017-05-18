## Starting clean for Part 5 - Automating getting list of s&p 500
import bs4 as bs
import pickle # Pickle is used to serialize any python object (save any object, e.g. variable) Here will save sp500 list to not hit wiki again and again
import requests
# imported in part 5
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
# imported in part 8
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
# Import in part 11
from collections import Counter
from pprint import pprint
# Import in part 12:
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier # Voting class because using many classifiers

style.use('ggplot')

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
        # print(ticker)
        if not os.path.exists("stocks_dfs/{}.csv".format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
            except:
                df = web.DataReader(ticker.replace('.','-'), 'yahoo', start, end)
                # print("**** WARNING **** COULD NOT GET THE DATA FOR {}".ticker)
            df.to_csv("stocks_dfs/{}.csv".format(ticker))
        else:
            print("Alread have {}".format(ticker))


# get_data_from_yahoo()

##################
##### Part 7 #####
##################
## Combine all the tables together into one dataframe
## Take the adjusted close for all stocks and put it all together

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    # Go with the latest updated ticker file.

    for count, ticker in enumerate(tickers):
        df = pd.read_csv("stocks_dfs/{}.csv".format(ticker))
        df.set_index('Date', inplace = True)

        df.rename(columns = {'Adj Close': ticker}, inplace = True) # Column is now for example AAPL
        df.drop(['Open','High','Low','Close','Volume'], axis = 1, inplace = True) # Adjusted close is now ticker

        # Start joining all the dataframes
        if main_df.empty:
            main_df = df
        else: # Already has a df
            main_df = main_df.join(df, how = 'outer') # Keep both rows (if one stock has it and the other not)

        if count % 10 == 0:
            print(count) # Cool, I always do this :) - yea sentdex

    main_df.to_csv("sp500_joined_closes.csv")

##################
##### Part 8 #####
##################

"""
Typically, would analyze data over specific timeframes,
not 17 year period. Doing this for example sake
"""

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr() # Create correlation table of our dataframe
    """
    @StockNotes:
    - Mean reversion: take 2 stocks that are highly correlated
    one starts to deviate, short one and invest in the other one
    eventually they come back together
    - Or if negatively correlated, same
    - Neutrally correlated and want to be diversified, invest in none
    correlated stocks
    """

    data = df_corr.values # Np array of columns and rows (no headers and index)
    fig = plt.figure()
    ax = fig.add_subplot(111) # 1 by 1, plot # 1 - or add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn) # plot colors, from red to yellow
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + .5, minor = False)
    ax.set_yticks(np.arange(data.shape[1]) + .5, minor = False)
    ax.invert_yaxis # Do this because a gap at the top of most matplotlib maps
    ax.xaxis.tick_top() # Put ticks on top instead of bottom

    column_labels = df_corr.columns
    row_labels = df_corr.index # They just be identical (col and row label)

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation = 90)
    heatmap.set_clim(-1,1) # limit of these colors, -1 is the min and 1 is the max
    # # For covariance, don't bound the scale!
    plt.tight_layout()
    plt.show()
    """
    For the above, for heatmap, no official heatmap
    Take the colors and plot them on the grid
    Add ticks that you can mark but also add company labels
    => Arrange labels at every half-mark (1.5,2.5...)
    """


##################
##### Part 9 #####
##################

"""
@stockNotes
Question is: with pricing data, there are relationships between companies
Can we get a machine to recognize relationships
Hypothesis: Group of companies are likely to move up or down, but may not move at
the same time (lead or lagging)
Lots of people try to chart patterns for one company, to predict if company will go anywhere
Question is: use just pricing data but take into account mvment of all companies + company in question

Convert pricing data to % change and will be our features
Target will be buy, sell or hold
Take all the feature data, generate label on training data
"Did the price, within the next 7 trading, go up by more than 2%
If it fell by 2% or more, sell, otherwise hold"
"""

def process_data_for_labels(ticker):
    """
    @argument: A specific ticker
    @returns:
        - tickers: all tickers
        - df: list with percent change of stocks (next 7 days) for specified ticker
            + all the other ticker info
    """
    ## Each generated model will be on a per company basis
    # Each company will take into account pricing of all other companies in sp500
    hm_days = 7 # We have 7 days to make or lose x %
    df = pd.read_csv("sp500_joined_closes.csv", index_col = 0)
    tickers = df.columns.values.tolist() # .values works probably
    df.fillna(0, inplace = True)

    for i in range(1, hm_days + 1): # Go all the way to 7
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        # rounded_Pchange = round((df[ticker].shift(-i) - df[ticker]) / df[ticker],2)
        # df['{}_{}d'.format(ticker, i)] = rounded_Pchange # EXON_2d (day 2 into the future) - shift(-i) is i days in future

    # @Note: looking at correlation on years of data - don't want to do that, relationships change over time
    # Look one or two year in the past (need more than daily!)

    df.fillna(0, inplace = True) # will not have nas anyway, because the value will be replaced by 0!
    return tickers, df


###################
##### Part 10 #####
###################

"""
Writing the function for target
Is this company a buy / sell / hold

for mapping tutiorial, look at pythonprogramming.net
for args, search also in pythonprog - allows to pass any number of arguments
"""

def buy_sell_hold(*args):
    """
    @arguments: takes different columns from df with Pchanges
    returns 1,-1 or 0 based on the condition
    """
    cols = [c for c in args]
    # print("COLS: ",cols)
    # With mapping to pandas, got a function that goes by row and can pass nothing or cols as parameter
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement: #### THIS IS WHERE UR @ERROR WAS, because of this line, never went to 0!
            return -1

    return 0 ## @Note: Will return 0 if number is a nan (happens for the tail of the dataframe)

###################
##### Part 11 #####
###################

"""
Map the function buy_sell_hold
to a new column
"""
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    # Define the new column, mapped answer of buy_sell_hold
    # args will be 7 day % changes

    ### @Note: What is being passed here: the percent change (for every day, for every hm_days)
    ## for 2d,3d,4d, will se nan in the tail of df, these will be returned as 0 by buy_sell_hold
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                                df['{}_1d'.format(ticker)],
                                                df['{}_2d'.format(ticker)],
                                                df['{}_3d'.format(ticker)],
                                                df['{}_4d'.format(ticker)],
                                                df['{}_5d'.format(ticker)],
                                                df['{}_6d'.format(ticker)],
                                                df['{}_7d'.format(ticker)]))

#################### FOR TESTING AND UNDERSTANDING CODE ####################
    # test = pd.DataFrame() ### TESTING TO TRY AND UNDERSTAND ZIS.
    # test['{}_target'.format(ticker)] = [buy_sell_hold(x) for x in df['{}_1d'.format(ticker)].tail()]
    # ## Uncomment: used for testing
    # df['{}_target'.format(ticker)] = test['{}_target'.format(ticker)]
    # print("TICKER:", df[ticker].tail(), "\n")
    # # print("MAPPED RES ",df['{}_target'.format(ticker)].tail(), "\n")
    # print("MAPPED RES \n ",test['{}_target'.format(ticker)], "\n")
    # # for i in range(1,8):
    # print("PCHANGE{}d:".format(1), df['{}_1d'.format(ticker)].tail(),"\n")
#################### END OF TESTING FOR UNDERSTANDING CODE ####################

    vals = df['{}_target'.format(ticker)].values.tolist()

    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals)) # Counter needs strings?

    """
    @StockNotes
    With random choices, should get 33% accuracy (not true)
    Usually, stocks go up in price
    Should probably change requirements
    Benchmark should be constant buy, ml algo should beat this
    """

    df.fillna(0, inplace = True) # np -> na
    df = df.replace([np.inf, -np.inf], np.nan) # percent change from 0 to a number
    df.dropna(inplace = True)

    # Create the feature set and labels
    df_vals = df[[ticker for ticker in tickers]].pct_change() # tickers is just the prices, normalized
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace = True)

    # Features and labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df # Return features, labels and df.


def do_ml(ticker):
    # Reminer: X is % change for specific company + FOR ALL COMPANIES
        # y is the target for the specified company (-1,1,0)
    X, y, df = extract_featuresets(ticker)

    # Create training and testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                        y,
                                                                        test_size = .25)
    # clf = neighbors.KNeighborsClassifier()

    ### BUT LET"S NOT JUST USE ONE CLASSIFIER (knn here)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])  ## Done without parameter tuning.

    clf.fit(X_train, y_train) # Can pickle the trained model
    confidence = clf.score(X_test, y_test) # If happy with prediction, go ahead and predict below
    print("Accuracy", confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))

    return confidence

do_ml('BAC')
