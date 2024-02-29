import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv1D, MaxPooling1D


# data read
data = pd.read_csv('SP500.csv', low_memory=False)
name = 'S&P500'

test_size = 100
pred_size = 10
valid_size = 0.2
np.random.seed(10)

t = data['Close']

test_date = data[-test_size+11:].set_index('Date').index
#test_date = np.append(test_date, datetime.datetime.now().strftime("%m/%d/%Y"))
test_date = np.append(test_date, "04/22/2021")
#date = data.index

y = preprocessing.MinMaxScaler()
yt =  y.fit_transform(data[['Close']])


data = preprocessing.MinMaxScaler().fit_transform(data[['Close', 'Open', 'High', 'Low']])

data = pd.DataFrame(data)
data.columns = ['Close', 'Open', 'High', 'Low']



# get dataset based on window_size(number of days to combine)
def make_dataset(data, label, window_size):
    data_list = []
    label_list = []
    for i in range(len(data) - window_size):
        data_list.append(np.array(data.iloc[i : i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(data_list), np.array(label_list)



# preprocessing training & validation
train = data[:-test_size]   # from start to test_size days before from now
train_data = train[:-1]
train_label = train['Close'].shift(-1)[:-1]
train_data, train_label = make_dataset(train_data, train_label, pred_size)
x_train, x_valid, y_train, y_valid = train_test_split(train_data, train_label, test_size=valid_size)

# preprocessing test data
test = data[-test_size:]    # from test_size days before to now
test_data = test
test_label = test['Close'].shift(-1)
test_data, test_label = make_dataset(test_data, test_label, pred_size)



# CNN model
model = Sequential()

model.add(Conv1D(
    filters=32, 
    kernel_size=5, 
    activation='relu',
    input_shape=(x_train.shape[1], x_train.shape[2])
))

model.add(MaxPooling1D(pool_size=2))

# flatten channels
model.add(Flatten())

# use 1 neural network
model.add(Dense(1, activation='relu'))



# learning the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# early stop when same lowest val_loss appeard 100 times
early_stop = EarlyStopping(monitor='val_loss', patience=100)     

# save the model
filename = os.path.join('./model/', 'tmp_checkpoint.h5')

# set best result as checkpoint
checkpoint = ModelCheckpoint(filename, monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='auto')

model.fit(x_train, y_train, epochs=10000,
                    validation_data=(x_valid, y_valid),
                    batch_size=len(train_data) // 4,
                    callbacks=[early_stop, checkpoint])

# load weights from best result
model.load_weights(filename)



pred = pd.DataFrame(model.predict(test_data))

#prediction.to_csv('LSTM_result/'+name+'.csv')
p = y.inverse_transform(pred)
test_label = test_label.reshape(-1,1)
d = y.inverse_transform(test_label)

plt.plot(test_date, d, color = 'red', label = 'Real '+name+' Stock Price')
plt.plot(test_date, p, color = 'blue', label = 'Predicted '+name+' Stock Price')
plt.title(name + ' Stock Price Prediction')
plt.xlabel('Time')
plt.xticks(ticks=test_date, rotation=45)
plt.locator_params(axis='x', nbins=10)
plt.ylabel('Stock Price')
plt.legend()
plt.show()



#print("=====Tomorrow(" + datetime.datetime.now().date().strftime("%m/%d/%Y") + ") prediction=====")
print("=====Tomorrow(" + test_date[-1:] + ") prediction=====")
print(p[-1:])
MAE = metrics.mean_absolute_error(d[:-1], p[:-1])
SMSE = np.sqrt(metrics.mean_squared_error(d[:-1], p[:-1]))
mean = np.mean(d[:-1])
print("Mean absolute error: " + str(round(MAE,3)))
print("Mean absolute error percentage: " + str(round((MAE/mean)*100, 3)) + "%")
print("Sqrt of mean squared error: " + str(round(SMSE,3)))
print("Sqrt of mean squared error percentage: " + str(round((SMSE/mean)*100,3)) + "%")