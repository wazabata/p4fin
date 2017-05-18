## Starting clean for Part 5 - Automating getting list of s&p 500
import bs4 as bs
import pickle # Pickle is used to serialize any python object (save any object, e.g. variable) Here will save sp500 list to not hit wiki again and again
import requests
#
# def save_sp500_tickers():
#     """
#     Get the source code for wikipedia
#     """
#     resp = requests.get('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')
#     soup = bs.BeautifulSoup(resp.text, "lxml") # resp.txt is text of source code
#     table = soup.find('table', {'class':'wikitable sortable'})
#     tickers = []
#     for row in table.findAll('tr')[1:]:
#         ticker = row.findAll('td')[0].text # Shouldn't it be [1]
#         tickers.append(ticker)
#
#     with open("sp500tickers.pickle", 'wb') as f: # wb = write byte
#         pickle.dump(tickers, f)
#
#     print(tickers)
#
#     return tickers
#
# save_sp500_tickers()
