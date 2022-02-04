import yfinance as yf
df = yf.download(tickers='BTC-USD', period="730d",interval="1h")
df = df.drop('Adj Close', 1) 
df.to_csv('../financial_data/BTC-USD.csv', header=False)
