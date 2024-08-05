# %%
#Importing necessary librabries
from dotenv import load_dotenv
import os 
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import hopsworks
import re 
#prepocessing
import requests
import pandas as pd
import json
#import pandas_market_calendars as mcal
import datetime
import numpy as np
from datetime import timedelta
load_dotenv()   #Making the .env file work

# %%
#Setting up API key to being able to fetch stocks from Alpha Vantage

api_key = os.getenv('STOCK_API') 
ts = TimeSeries(key=api_key, output_format='pandas')

#Defining a function to fetch stocks

def fetch_stock_prices(symbol):
    # Fetch daily adjusted stock prices; adjust the symbol as needed
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    
    # Add a new column named 'ticker' and fill it with the ticker name
    data['ticker'] = symbol
    
    return data

#Usage
symbol = 'NVDA'
stock_data = fetch_stock_prices(symbol)
print(stock_data.head())

# %%
# Defining the file path and name
file_path = 'NVDA_stock_prices.csv'  

# Saving the DataFrame to CSV
stock_data.to_csv(file_path)

print(f"Data saved to {file_path}")