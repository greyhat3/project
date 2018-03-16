import bs4 as bs
import datetime
#from datetime import datetime 
import pandas as pd
import quandl
import os
import requests
import pickle
import pandas_datareader.data as web
quandl.ApiConfig.api_key = 'xFzMGMyH78RLpYF6yHzs'

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
   
    soup= bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    return tickers

#save_sp500_tickers()

def get_data_from_quandl(reload_sp500=False):
       
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('adj_dfs'):
        os.makedirs('adj_dfs')

    start = datetime.datetime(2000,1,1) 
    end = datetime.datetime.now()

    for ticker in tickers[:20]:
        print(ticker)
        if not os.path.exists('adj_dfs/{}.csv'.format(ticker)):
            data = quandl.get_table('WIKI/PRICES', qopts = { 'columns': ['ticker', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume', 'adj_close'] }, date = { 'gte': '2017-11-11', 'lte': '2018-01-01' } , paginate=True)
         
           # df = web.DataReader(ticker,'yahoo', start, end)
            data.to_csv('adj_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

get_data_from_quandl()
