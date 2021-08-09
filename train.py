import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("fivethirtyeight")
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from sklearn.metrics import mean_squared_error
import yfinance as yf
import math
    
#tickers = ['INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN', 'NFLX', 'FB', 'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK', 'TSLA', 'MSFT', 'SNPS']
tickers = ['NVDA']
"""
df_list = list()
for ticker in tickers:
    data = yf.download(ticker, group_by="Ticker")
    df_list.append(data)
#print(df_list)

# save to csv
for index, ticker in enumerate(tickers):
    df_list[index].to_csv("data/" + ticker + ".csv")
"""    
# training...    
for ticker in tickers:
    df = pd.read_csv("data/" + ticker + ".csv", index_col="Date", parse_dates=["Date"])
    #print(data.shape)
    df = df.dropna()
    #print(pd.isna(df).sum())
    
    """
    #plot adj closing history
    plt.figure(figsize=(16, 8))
    plt.title("Adj Closing Price History")
    plt.plot(data["Adj Close"])
    plt.xlabel("Date")
    plt.ylabel("Adj Close Price")
    plt.show()
    """
    data = df.filter(items=["Adj Close"])
    dataset = data.values
    train_len = math.ceil(len(dataset)*.8)
    #print(train_len)
    
    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset) #should split first
    
    # create training dataset
    train_data = scaled_data[0:train_len, :]
    test_data = scaled_data[train_len-60:, :]
    #print(train_data[1])
    x_train = []
    y_train = []
    for i in range(60, len(train_data)): #append last 60 days values
        x_train.append(train_data[i-60:i, 0]) 
        y_train.append(train_data[i, 0])
        
    x_test = []
    y_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # convert to numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test = np.array(x_test)
    #print(x_train.shape, y_train.shape)
    
    # reshape array to fit into model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #print(x_train.shape)
    
    # The GRU architecture with Dropout regularization
    regressorGRU = Sequential()
    # First layer 
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Second layer
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Third layer
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth layer
    regressorGRU.add(GRU(units=50, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=1))
    # Compiling the RNN
    regressorGRU.compile(optimizer='adam',loss='mean_squared_error')
    # Training the RNN
    print("Fitting into " + ticker)
    regressorGRU.fit(x_train, y_train,epochs=10,batch_size=150)
    
    pred = regressorGRU.predict(x_test)
    pred = scaler.inverse_transform(pred)
     
    # Plot the result
    train = data[:train_len]
    valid = data[train_len:]
    valid["Pred"] = pred
    plt.figure(figsize=(16, 8))
    plt.title(ticker)
    plt.xlabel("Date")
    plt.ylabel("Adj Close Price")
    plt.plot(train["Adj Close"])
    plt.plot(valid[["Adj Close", "Pred"]])
    plt.legend(["Train", "Real", "Predict"])
    plt.show()
    
    # save model
    filename = ticker + "_regressor.h5"  
    regressorGRU.save("saved_models/" + filename)
