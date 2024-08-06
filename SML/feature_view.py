# %%
# Importing necessary libraries
import pandas as pd               # For data manipulation using DataFrames
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For data visualization
import os                         # For operating system-related tasks
import joblib                     # For saving and loading models
import hopsworks                  # For getting access to hopsworks

from feature_pipeline import nvidia_fg   #Loading in the tesla_fg
from feature_pipeline import news_sentiment_fg  #Loading in the news_fg

#Making the notebook able to fetch from the .env file
from dotenv import load_dotenv
import os

load_dotenv()

#Getting connected to hopsworks
api_key = os.environ.get('HOPSWORKS_API')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# %%
#Defining the function to create feature view

def create_stocks_feature_view(fs, version):

    # Loading in the feature groups
    nvidia_fg = fs.get_feature_group('nvidia_stock', version=1)
    news_sentiment_fg = fs.get_feature_group('news_sentiment_updated', version=1)

    # Defining the query
    ds_query = nvidia_fg.select(['date', 'open', 'ticker'])\
        .join(news_sentiment_fg.select(['sentiment']))

    # Creating the feature view
    feature_view = fs.create_feature_view(
        name='nvidia_stocks_fv',
        query=ds_query,
        labels=['open']
    )

    return feature_view, nvidia_fg

# %%
#Creating the feature view
try:
    feature_view = fs.get_feature_view("nvidia_stocks_fv", version=1)
    nvidia_fg = fs.get_feature_group('nvidia_stock', version=1)
except:
    feature_view, nvidia_fg = create_stocks_feature_view(fs, 1)
