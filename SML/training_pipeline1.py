# %%
#Importing necessary libraries
import hopsworks
import hsfs
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler  # Import StandardScaler from scikit-learn
import joblib

load_dotenv()

#Connecting to hopsworks
api_key = os.environ.get('hopsworks_api')
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

#Another connection to hopsworks
api_key = os.getenv('hopsworks_api')
connection = hsfs.connection()
fs = connection.get_feature_store()

# %%
#Getting the feature view
feature_view = fs.get_feature_view(
    name='nvidia_stocks_fv',
    version=1
)

# %%
#Setting up train & test split dates
train_start = "2022-06-22"
train_end = "2023-12-31"

test_start = '2024-01-01'
test_end = "2024-05-03"

# %%
#Creating the train/test split on the feature view with the split dates
feature_view.create_train_test_split(
    train_start=train_start,
    train_end=train_end,
    test_start=test_start,
    test_end=test_end,
    data_format='csv',
    coalesce= True,
    statistics_config={'histogram':True,'correlations':True})

# %%
#Collecting the split from feature view
X_train, X_test, y_train, y_test = feature_view.get_train_test_split(1)

# %%
#Inspecting X_train
X_train

# %%
#Converting date into datetime
X_train['date'] = pd.to_datetime(X_train['date']).dt.date
X_test['date'] = pd.to_datetime(X_test['date']).dt.date
X_train['date'] = pd.to_datetime(X_train['date'])
X_test['date'] = pd.to_datetime(X_test['date'])

# %%
X_train.head()

# %%
# Extracting the 'ticker' column
tickers = X_train[['ticker']]

# Initializing OneHotEncoder
encoder = OneHotEncoder()

# Fitting and transforming the 'ticker' column
ticker_encoded = encoder.fit_transform(tickers)

# Converting the encoded column into a DataFrame
ticker_encoded_df = pd.DataFrame(ticker_encoded.toarray(), columns=encoder.get_feature_names_out(['ticker']))

# Concatenating the encoded DataFrame with the original DataFrame
X_train = pd.concat([X_train, ticker_encoded_df], axis=1)

# Dropping the original 'ticker' column
X_train.drop('ticker', axis=1, inplace=True)

# %%
#Inspecting X train after onehotencoding 'Ticker'
X_train.head()

# %%
#Doing the same for X test as done to X train

tickers = X_test[['ticker']]

# Initializing OneHotEncoder
encoder = OneHotEncoder()

# Fitting and transforming the 'ticker' column
ticker_encoded_test = encoder.fit_transform(tickers)

# Converting the encoded column into a DataFrame
ticker_encoded_df_test = pd.DataFrame(ticker_encoded_test.toarray(), columns=encoder.get_feature_names_out(['ticker']))

# Concatenating the encoded DataFrame with the original DataFrame
X_test = pd.concat([X_test, ticker_encoded_df_test], axis=1)

# Dropping the original 'ticker' column
X_test.drop('ticker', axis=1, inplace=True)

# %%
#Loading in MinMaxScaler to be used on the target variable 'open'
scaler = MinMaxScaler()

# Fitting and transforming the 'open' column
#y_train['open_scaled'] = scaler.fit_transform(y_train[['open']])
#y_train.drop('open', axis=1, inplace=True)

# %%
#Doing the same to y_test as done to y_train 
#y_test['open_scaled'] = scaler.fit_transform(y_test[['open']])
#y_test.drop('open', axis=1, inplace=True)

# %%
#Defining the function for the LSTM model
def create_model(input_shape,
                 LSTM_filters=64,
                 dropout=0.1,
                 recurrent_dropout=0.1,
                 dense_dropout=0.5,
                 activation='relu',
                 depth=1):

    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    if depth > 1:
        for i in range(1, depth):
            # Recurrent layer
            model.add(LSTM(LSTM_filters, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))

    # Recurrent layer
    model.add(LSTM(LSTM_filters, return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout))

    # Fully connected layer
    if activation == 'relu':
        model.add(Dense(LSTM_filters, activation='relu'))
    elif activation == 'leaky_relu':
        model.add(Dense(LSTM_filters))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

    # Dropout for regularization
    model.add(Dropout(dense_dropout))

    # Output layer for predicting one day forward
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model

# %%
# As X_train['date'] column exists and is in datetime format, we're converting it
X_train['year'] = X_train['date'].dt.year
X_train['month'] = X_train['date'].dt.month
X_train['day'] = X_train['date'].dt.day

# Dropping the original date column
X_train.drop(columns=['date'], inplace=True)

# Converting dataframe to numpy array
X_train_array = X_train.to_numpy()

# Reshaping the array to have a shape suitable for LSTM
X_train_array = np.expand_dims(X_train_array, axis=1)

# %%
# Convert DataFrame to numpy array
X_train_array = X_train.values

# Reshaping X_train_array to add a time step dimension
X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], 1, X_train_array.shape[1])

# Assuming X_train_reshaped shape is now (374, 1, 5)
input_shape = X_train_reshaped.shape[1:]

# Create the model
model = create_model(input_shape=input_shape)

# %%
#Fitting the model on the training dataset
model.fit(X_train_reshaped, y_train)

# %%
# As X_test['date'] column exists and is in datetime format, we're converting it
X_test['year'] = X_test['date'].dt.year
X_test['month'] = X_test['date'].dt.month
X_test['day'] = X_test['date'].dt.day

# Dropping the original date column
X_test.drop(columns=['date'], inplace=True)

# Converting dataframe to numpy array
X_test_array = X_test.to_numpy()

# Reshape the array to have a shape suitable for LSTM
X_test_array = np.expand_dims(X_test_array, axis=1)

# %%
# Predicting y_pred with X_test
y_pred = model.predict(X_test_array)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)

# %%
#Conneting to hopsworks model registry
mr = project.get_model_registry()

# %%
# Compute RMSE metric for filling the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_metrics = {"RMSE": rmse}
rmse_metrics

# %%
#Setting up the model schema
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

# %%
#Creating a file colled 'stock_model'
model_dir="stock_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

# %%
#Saving the model to hopsworks model registry
#stock_pred_model = mr.tensorflow.create_model(
#        name="stock_pred_model",
#        metrics= rmse_metrics,
#        model_schema=model_schema,
#        description="Stock Market TSLA Predictor from News Sentiment",
#    )

#stock_pred_model.save(model_dir)

# %%
def register_tensorflow_model(model, name, description, features, labels, metrics):
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import os
    import joblib
    import shutil

    mr = project.get_model_registry()

    model_dir= name + "_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    pickle= name + '_model.pkl'
    # This will strip out the sml directory, copying only the files
    #shutil.copytree("sml", model_dir, dirs_exist_ok=True) #python 3.8+

    joblib.dump(model, model_dir + "/" + pickle)

    input_example = features.sample()
    input_schema = Schema(features)
    output_schema = Schema(labels)
    model_schema = ModelSchema(input_schema, output_schema)

    stock_pred_model = mr.tensorflow.create_model(
        name="stock_pred_model", 
        metrics=rmse_metrics,
        model_schema=model_schema,
        input_example=input_example, 
        description=description)

    # Save all artifacts in the model directory to the model registry
    stock_pred_model.save(model_dir)


register_tensorflow_model(model, "stock_prediction", "Stock Prediction", X_train, y_train, rmse_metrics)
