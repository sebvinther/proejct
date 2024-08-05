# %%
#Importing necessary libraries
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
import os
import time
import pandas as pd 
from textblob import TextBlob

# %%
#Defining a function to process news articles
def process_news_articles(news_articles):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(news_articles)

    # Drop rows where the description is NaN
    df = df.dropna(subset=['description'])

    # Fill missing 'amp_url' and 'keywords' with specific placeholders
    df['amp_url'] = df['amp_url'].fillna('No URL provided')
    df['keywords'] = df['keywords'].fillna('No keywords')

    # Sentiment analysis on descriptions
    df['sentiment'] = df['description'].apply(lambda text: TextBlob(text).sentiment.polarity)

    # Convert 'published_utc' to datetime and extract date and time
    df['published_utc'] = pd.to_datetime(df['published_utc'])
    df['date'] = df['published_utc'].dt.date
    df['time'] = df['published_utc'].dt.time

    # Dropping unnecessary columns
    df.drop(['published_utc'], axis=1, inplace=True)
    # set date to index
    df = df.set_index("date")
    df.reset_index(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.groupby(['date', 'ticker'])['sentiment'].mean().reset_index()

    return df

# %%
#Defining a function for the exponential moving average

def exponential_moving_average(df, window):
   # Calculate EMA on the 'sentiment' column
    df[f'exp_mean_{window}_days'] = df['sentiment'].ewm(span=window, adjust=False).mean()
    return df