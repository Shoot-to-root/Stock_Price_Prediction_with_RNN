import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from datetime import datetime

tickers = ['INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN', 'NFLX', 'FB', 'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK', 'TSLA', 'MSFT', 'SNPS']
date = datetime.today().strftime('%m%d')
#print(date)
rate_of_change_list = []

for ticker in tickers:
    df = pd.read_csv("data/" + ticker + ".csv", index_col="Date", parse_dates=["Date"])
    #print(data)
    data = df.filter(items=["Adj Close"])
    #print(data.tail())
    dataset = data.values
    
    # Get last 60 days
    last60 = data[-60:]
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(dataset)
    scaled_last60 = scaler.transform(last60)
    #print(scaled_last60.shape)
    
    x_test = []
    x_test.append(scaled_last60)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # load model
    loaded_model = tf.keras.models.load_model("saved_models/" + ticker + "_regressor.h5")
    #loaded_model.summary()
    
    # predict
    yesterday_close_price = dataset[-1:]
    result = loaded_model.predict(x_test)
    result = scaler.inverse_transform(result)
    print(ticker + " predicted adj close price: " + str(result))
    print(ticker + " yesterday adj close price: " + str(yesterday_close_price))
    
    rate_of_change = abs(result - yesterday_close_price) / yesterday_close_price
    rate_of_change = rate_of_change * 100
    #print(rate_of_change)
    
    if result > dataset[-1:] and rate_of_change > 1.5:
        rate_of_change_list.append(0)
        print(0)
    elif result < dataset[-1:] and rate_of_change > 1.5:
        rate_of_change_list.append(2)
        print(2)
    elif rate_of_change <= 1.5:
        rate_of_change_list.append(1)
        print(1)
    else:
        print("Something's wrong!")
        exit(0)

with open("pred/" + date + "_prediction.txt", "w") as f:
        for i in rate_of_change_list:
            f.write("%s\n" % i)