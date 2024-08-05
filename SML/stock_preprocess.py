# %%
#Importing necessary libraries
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
import pandas_market_calendars as mcal
import datetime
import numpy as np
from datetime import datetime, timedelta
load_dotenv()

# %%
#Connecting to Alpha vantage using API key
api_key = os.getenv('STOCK_API') # Replace this with your actual API key
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch daily adjusted stock prices; adjust the symbol as needed
data, meta_data = ts.get_daily(symbol='NVDA', outputsize='full')

print(data.head())

# %%
#Looking at data info
data.info()

# %%
#Looking at the meta data
meta_data

# %%
#Stock market:
def today_is_a_business_day(today):
    # Get the NYSE calendar
    cal = mcal.get_calendar('NYSE')
    schedule = cal.schedule(start_date=today, end_date=today) # Get the NYSE calendar's open and close times for the specified period
    try:
        isBusinessDay = schedule.market_open.dt.strftime('%Y-%m-%d')
        return True
    except:
        print('Today {} is not a business day'.format(today))
        return False

# %%
#Defining a function to find the next business day
def next_business_day(today):
    
    # Real tomorrow
    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Get the NYSE calendar
    cal = mcal.get_calendar('NYSE')

    found_next_business_day = False 
    while not found_next_business_day:
        schedule = cal.schedule(start_date=tomorrow, end_date=tomorrow) # Get the NYSE calendar's open and close times for the specified period
        try:
            isBusinessDay = schedule.market_open.dt.strftime('%Y-%m-%d') # Only need a list of dates when it's open (not open and close times)
            found_next_business_day = True
        except:
            print('The date {} is not a business day'.format(tomorrow))
            tomorrow = (datetime.datetime.strptime(tomorrow,"%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            
    return isBusinessDay.to_numpy()[0]

# %%
#Defining a function to extract business day
def extract_business_day(start_date,end_date):
    """
    Given a start_date and end_date.
    
    `Returns`:
    
    isBusinessDay: list of str (with all dates being business days)
    is_open: boolean list
        e.g is_open = [1,0,...,1] means that start_date = open, day after start_date = closed, and end_date = open
    """
    
    # Saving for later
    end_date_save = end_date
    
    # Getting the NYSE calendar
    cal = mcal.get_calendar('NYSE')

    # Getting the NYSE calendar's open and close times for the specified period
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    
    # Only need a list of dates when it's open (not open and close times)
    isBusinessDay = np.array(schedule.market_open.dt.strftime('%Y-%m-%d')) 
    
    # Going over all days: 
    delta = datetime.timedelta(days=1)
    start_date = datetime.datetime.strptime(start_date,"%Y-%m-%d") #datetime.date(2015, 7, 16)
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d") #datetime.date(2023, 1, 4)
    
    # Extracting days from the timedelta object
    num_days = (end_date - start_date).days + 1
    
    # Creating a boolean array for days being open (1) and closed (0) 
    is_open = np.zeros(num_days)
    
    # iterate over range of dates
    current_BusinessDay = isBusinessDay[0]
    count_dates = 0
    next_BusinessDay = 0
    
    while (start_date <= end_date):
    
        if start_date.strftime('%Y-%m-%d') == current_BusinessDay:
            is_open[count_dates] = True

            if current_BusinessDay == end_date_save or current_BusinessDay==isBusinessDay[-1]:
                break
            else:
                next_BusinessDay += 1
                current_BusinessDay = isBusinessDay[next_BusinessDay]
        else:
            is_open[count_dates] = False

        count_dates += 1   
        start_date += delta
        
    print(np.shape(is_open))
        
    return isBusinessDay, is_open

# %%
#Defining a function to clean the column names
def clean_column_name(name):
    # Remove all non-letter characters
    cleaned_name = re.sub(r'[^a-zA-Z]', '', name)
    return cleaned_name

# %%
data.columns = [clean_column_name(col) for col in data.columns]

# %%
data.head()

# %%
data.reset_index(inplace=True)


# %%
data.head()

# %%
# Define the date range we're interested in
yesterday =datetime.now()-timedelta(days=1)
two_years_back = yesterday - timedelta(days=684)

# %%
# Filtering the DataFrame to this range
filtered_df = data[(data['date'] >= two_years_back) & (data['date'] <= yesterday)]

# %%
filtered_df.head()

# %%
print(filtered_df['date'].min())
print(filtered_df['date'].max())

# %%
filtered_df.shape