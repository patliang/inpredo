# Imports
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si 
from datetime import datetime, timedelta, date
import threading    
import yfinance as yf
import os
import pandas as pd
from subprocess import Popen

yf.pdr_override()

csv_down_dir = 'csv/db_yfinance/'
csv_screen_dir = 'stock_vcpscreener/db_yfinance/'

lastest_date = ''

if not os.path.exists(csv_down_dir):
    os.makedirs(csv_down_dir)
    
# Variables 
def get_csv():
    tickers = si.tickers_nasdaq() + si.tickers_other()
    tickers.sort()
    tickers = [item.replace(".", "-") for item in tickers] # Yahoo Finance uses dashes instead of dots 
    exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", 
                                        "200 Day MA", "52 Week Low", "52 week High"]) 
    return tickers

def get_ticker(tickers):
    #start_date = datetime.now() - timedelta(days=3650)
    #end_date = date.today()
    lastest_date = ''

    # Find top 30% performing stocks (relative to the S&P 500)
    for ticker in tickers:
        try:
            # Download historical data as CSV for each stock (makes the process faster) 
            #df = pdr.get_data_yahoo(ticker, start_date, end_date) 
            df = pdr.get_data_yahoo(ticker) 
            #df = pdr.get_data_yahoo(ticker) 
            df.to_csv(f'{csv_down_dir}'+ticker.strip().ljust(5,'_')+'.csv') 
            if (lastest_date == '') and (len(df)>1): 
                # Read in the database update date
                lastupdate = pd.read_csv(csv_screen_dir+"last_update.dat", header=0)
                lastupdate['Date']=pd.to_datetime(lastupdate['Date']) 

                # When done, update the last update file to current time (UTC-5). Now trade_day instead 
                lastupdate['Date'] = df.index[-1]
                lastupdate.to_csv(csv_screen_dir+"last_update.dat", mode='w', index=False) 
        except Exception as e:
            print (e)
            print(f"Error downloading data for {ticker}")
            pass
         
def update_screener_db(): 
    for filename in os.listdir(csv_down_dir):
        df_down = pd.read_csv(csv_down_dir+'{}'.format(filename))
        if os.path.exists(csv_screen_dir+filename):
            path_file = csv_screen_dir+'{}'.format(filename)
            df_screen = pd.read_csv(path_file)
            if len(df_down)>len(df_screen):
                df_down.to_csv(path_file)
                print(f'{filename} copied') 
    
# get list of stocks
tickers = get_csv()

# starting thread    
get_ticker(tickers) 

#update screener csv 
update_screener_db()


#do the screener scanning and reporting
os.system('python stock_vcpscreener.py') 
