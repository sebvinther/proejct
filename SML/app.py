import streamlit as st
import pandas as pd
from stocks import fetch_stock_prices
from news import fetch_news
from stock_preprocess import preprocess_stock_data
from news_preprocess import preprocess_news_data
from training_pipeline1 import load_model, predict_stock_prices

# Load environment variables (API keys, etc.)
from dotenv import load_dotenv
load_dotenv()

# Title and Introduction
st.title("Nvidia Stock Price Prediction")
st.write("Predict Nvidia's stock prices using machine learning and sentiment analysis from news articles.")

# Sidebar for Inputs
st.sidebar.header("User Input Parameters")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Fetch and Preprocess Data
if st.sidebar.button("Fetch Data"):
    # Fetch data
    stock_data = fetch_stock_prices('NVDA', start_date, end_date)
    news_data = fetch_news('NVDA', start_date, end_date)
    
    # Preprocess data
    stock_data_preprocessed = preprocess_stock_data(stock_data)
    news_data_preprocessed = preprocess_news_data(news_data)
    
    st.write("Data Fetched and Preprocessed Successfully!")
    st.write("Sample Stock Data:", stock_data_preprocessed.head())
    st.write("Sample News Data:", news_data_preprocessed.head())

# Load Model and Predict
if st.sidebar.button("Run Prediction"):
    # Load the pre-trained LSTM model
    model = load_model()  
    predictions = predict_stock_prices(model, stock_data_preprocessed, news_data_preprocessed)
    
    st.write("Predicted Stock Prices:")
    st.line_chart(predictions)

