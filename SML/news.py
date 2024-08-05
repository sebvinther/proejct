# %%
#Importing necessary libraries
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
import os
import time
import pandas as pd 
from news_preprocess import *    #Importing everything from 'news_preprocess'
load_dotenv()

# %%
#Defining a function for fetching news

def fetch_news(ticker, start_date, end_date):
    api_key = os.getenv('API_NEWS')
    base_url = os.getenv("ENDPOINTNEWSP")
    headers = {"Authorization": f"Bearer {api_key}"}
    all_news = []
    
    current_date = start_date

    while current_date <= end_date:
        batch_end_date = current_date + timedelta(days=50)
        if batch_end_date > end_date:
            batch_end_date = end_date

        params = {
            "ticker": ticker,
            "published_utc.gte": current_date.strftime('%Y-%m-%d'),
            "published_utc.lte": batch_end_date.strftime('%Y-%m-%d'),
            "limit": 50,
            "sort": "published_utc"
        }

        try:
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('results', [])
                
                # Creating a DataFrame from articles
                df = pd.DataFrame(articles)
                
                # Adding primary_key column if ticker is found
                df['ticker'] = df['tickers'].apply(lambda x: ticker if ticker in x else None)
                
                all_news.append(df)  # Append DataFrame to the list
                print(f"Fetched {len(articles)} articles from {current_date.strftime('%Y-%m-%d')} to {batch_end_date.strftime('%Y-%m-%d')}")
                current_date = batch_end_date + timedelta(days=1)
            elif response.status_code == 429:
                print("Rate limit reached. Waiting to retry...")
                time.sleep(60)  # Wait for 60 seconds or as recommended by the API
                continue  # Retry the current request
            else:
                print(f"Failed to fetch data: {response.status_code}, {response.text}")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return pd.concat(all_news, ignore_index=True)

#Usage
api_key = os.getenv('API_NEWS')
ticker = 'NVDA'
end_date = datetime.now() - timedelta(days=1)  # Yesterday's date
start_date = end_date - timedelta(days=365 * 2)
news_articles = fetch_news(ticker, start_date, end_date)
print(f"Total articles fetched: {len(news_articles)}")


# %%
# Process the news articles
df = process_news_articles(news_articles)

# %%
df.info()

# %%
df.head()

# %%
df= df.sort_index(ascending=False)

# %%
#Putting the news articles into a csv
df.to_csv('news_articles.csv', index=False)

# %%
df_processed = exponential_moving_average(df, window=7)

# %%
df_processed.to_csv('news_articles_ema.csv', index=False)

# %%
df_processed.head()

# %%
df_processed.tail()

# %%
print(df_processed['date'].min())
print(df_processed['date'].max())

# %%
print(df_processed['date'].max() - df_processed['date'].min()) 

# %%
df_processed.shape

# %%
duplicates = df_processed[df_processed.duplicated('date')]

# %%
duplicates.shape

# %%
df_processed.head()