# Imports
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si 
from datetime import datetime, timedelta, date
import threading    
import yfinance as yf
import os
import pandas as pd

yf.pdr_override()

csv_dir = 'csv/db_yfinance/'
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
    
# Variables 
def get_csv():
    tickers = si.tickers_nasdaq() + si.tickers_other()
    tickers.sort()
    tickers = [item.replace(".", "-") for item in tickers] # Yahoo Finance uses dashes instead of dots 
    exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", 
                                        "200 Day MA", "52 Week Low", "52 week High"]) 
    return tickers

def get_ticker(tickers, thread_no, start, end):
    #start_date = datetime.now() - timedelta(days=3650)
    #end_date = date.today()
    # Find top 30% performing stocks (relative to the S&P 500)
    for ticker in tickers[start:end]:
        try:
            # Download historical data as CSV for each stock (makes the process faster) 
            #df = pdr.get_data_yahoo(ticker, start_date, end_date) 
            df = pdr.get_data_yahoo(ticker) 
            df.to_csv(f'{csv_dir}'+ticker.strip().ljust(5,'_')+'.csv') 
        except Exception as e:
            print (e)
            print(f"Error downloading data for {ticker}")
            pass
        
# define thread  
def split_processing(items, num_splits=5):  
    split_size = len(items) // num_splits                                       
    threads = []                                                                
    for i in range(num_splits):                                                 
        # determine the indices of the list this thread will handle             
        start = i * split_size                                                  
        # special case on the last chunk to account for uneven splits           
        end = None if i+1 == num_splits else (i+1) * split_size                 
        # create the thread                                                     
        threads.append(                                                         
            threading.Thread(target=get_ticker, args=(items, str(i),  start, end)))         
        threads[-1].start() # start the thread we just created                  

    # wait for all threads to finish                                            
    for t in threads:                                                           
        t.join() 
 
# get list of stocks
tickers = get_csv()

# starting thread    
split_processing(tickers,5) 
