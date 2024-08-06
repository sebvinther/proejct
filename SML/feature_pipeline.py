# Import necessary libraries
import pandas as pd               # For data manipulation using DataFrames
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For data visualization
import os                         # For operating system-related tasks
import joblib                     # For saving and loading models
import hopsworks                  # For getting access to hopsworks
import re

# Import specific modules from scikit-learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder   # For data preprocessing
from sklearn.metrics import accuracy_score                        # For evaluating model accuracy

from dotenv import load_dotenv
import os
load_dotenv()

#Connecting to hopsworks
api_key = os.environ.get('hopsworks_api')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# %%
# Load and display the data from CSV to confirm
nvda_df = pd.read_csv('NVDA_stock_prices.csv')
print(nvda_df.head())    

# %%
#Defining a function to clean the column names
def clean_column_name(name):
    # Remove all non-letter characters
    cleaned_name = re.sub(r'[^a-zA-Z]', '', name)
    return cleaned_name

# %%
nvda_df

# %%
# Cleaning up column names for 'tsla_df'
nvda_df.columns = [clean_column_name(col) for col in nvda_df.columns]
print(nvda_df.columns)

# %%
# Converting the "date" column to timestamp
nvda_df['date'] = pd.to_datetime(nvda_df['date'])

# %%
# Defining the stocks feature group
nvidia_fg = fs.get_or_create_feature_group(
    name="nvidia_stock",
    description="Nvidia stock dataset from alpha vantage",
    version=1,
    primary_key=["ticker"],
    event_time=['date'],
    online_enabled=False,
)

# %%
#Inserting the stock data into the stocks feature group
nvidia_fg.insert(nvda_df, write_options={"wait_for_job" : False})

# %%
#Collecting news df
news_df = pd.read_csv('news_articles_ema.csv')

# %%
#Dropping exp mean 7 days
news_df_updated = news_df.drop(columns=['exp_mean_7_days'])

# %%
#Updating date to datetime
news_df_updated['date'] = pd.to_datetime(news_df_updated['date'])

# %%
#Defining the news feature group
news_sentiment_fg = fs.get_or_create_feature_group(
    name='news_sentiment_updated',
    description='News sentiment from Polygon',
    version=1,
    primary_key=['ticker'],
    event_time=['date'],
    online_enabled=False,
)

# %%
#Inserting the news data into the news feature group
news_sentiment_fg.insert(news_df_updated)
