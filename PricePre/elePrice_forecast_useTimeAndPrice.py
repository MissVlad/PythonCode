import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from matplotlib import pyplot as plt
from scipy.io import loadmat
import datetime as dt
import holidays

# 处理数据，形成可以用的完整数据集，数据集包括完整的时间信息，电价和买卖量信息。
uk_holidays = holidays.UnitedKingdom()
ele_datetime = pd.date_range(start='2017/1/1', end='2020/1/1', freq='H')
ele_Datetime = pd.DataFrame({'Date':ele_datetime})
DayofWeek = [int(i.strftime("%w")) for i in ele_datetime]
Holiday = [1 if str(val).split()[0] in uk_holidays else 0 for val in ele_Datetime['Date']]
ele_Datetime['Year'] = pd.DatetimeIndex(ele_Datetime['Date']).year
ele_Datetime['Month'] = pd.DatetimeIndex(ele_Datetime['Date']).month
ele_Datetime['Day'] = pd.DatetimeIndex(ele_Datetime['Date']).day
Year = ele_Datetime['Year'].values.tolist()
Month = ele_Datetime['Month'].values.tolist()
Day = ele_Datetime['Day'].values.tolist()

ele_data = loadmat('eleDayAhead_2017to19.mat')
ele_value = np.array(ele_data["eleDayAhead_17to19"].reshape(26281, 3))
ele_Value = {'Price': ele_value[:, 0], 'Buy': ele_value[:, 1], 'Sell': ele_value[:, 2]}
ele_value_withTime = DataFrame(ele_Value, index=ele_datetime)
BST1 = ele_value_withTime.loc['2017-03-26':'2017-10-28']
BST2 = ele_value_withTime.loc['2018-03-25':'2018-10-27']
BST3 = ele_value_withTime.loc['2019-03-31':'2019-10-26']
BST1['BST'] = '1'
BST2['BST'] = '1'
BST3['BST'] = '1'
noBST1 = ele_value_withTime.loc['2017-01-01':'2017-03-25']
noBST2 = ele_value_withTime.loc['2017-10-29':'2018-03-24']
noBST3 = ele_value_withTime.loc['2018-10-28':'2019-03-30']
noBST4 = ele_value_withTime.loc['2019-10-27':'2020-01-01']
noBST1['BST'] = '0'
noBST2['BST'] = '0'
noBST3['BST'] = '0'
noBST4['BST'] = '0'
data = noBST1.append(BST1).append(noBST2).append(BST2).append(noBST3).append(BST3).append(noBST4)
data['DayofWeek'] = DayofWeek
data['Holiday'] = Holiday
data['Year'] = Year
data['Month'] = Month
data['Day'] = Day
data = data[['Year','Month','Day','DayofWeek','Holiday','BST','Price','Buy','Sell']]
data['DayofWeek'] = data['DayofWeek'].replace(0,7)

# 数据集划分train和test集;简单例子-使用前一天电价和预测当天时间信息预测下一天的电价。
x_train_price = data.loc['2017-01-01':'2018-12-30',['Price']]
y_train_price = data.loc['2017-01-02':'2018-12-31',['Price']]
x_train_time = data.loc['2017-01-02':'2018-12-31',['Month','DayofWeek','Holiday','BST']]
x_test_price = data.loc['2018-12-31':'2019-12-30',['Price']]
y_test_price = data.loc['2019-01-01':'2019-12-31',['Price']]
x_test_time = data.loc['2019-01-01':'2019-12-31',['Month','DayofWeek','Holiday','BST']]

# 标准化数据集
# 使用one-hot标准化month和dayofweek
train_month_onehot = pd.get_dummies(x_train_time,columns=['Month']).iloc[:,3:]
train_dw_onehot = pd.get_dummies(x_train_time,columns=['DayofWeek']).iloc[:,3:]
test_month_onehot = pd.get_dummies(x_test_time,columns=['Month']).iloc[:,3:]
test_dw_onehot = pd.get_dummies(x_test_time,columns=['DayofWeek']).iloc[:,3:]
train_bstholi = x_train_time.iloc[:,2:]
Xtrain_time = pd.concat([train_month_onehot,train_dw_onehot,train_bstholi],axis=1)
test_bstholi = x_test_time.iloc[:,2:]
Xtest_time = pd.concat([test_month_onehot,test_dw_onehot,test_bstholi],axis=1)
#使用min-max标准化电价
min_price = np.min(x_train_price['Price'])
max_price = np.max(x_train_price['Price'])
x_train_price_normalise = x_train_price.sub(1.57).div(189.98)
y_train_price_normalise = y_train_price.sub(1.57).div(189.98)
x_test_price_normalise = x_test_price.sub(1.57).div(189.98)
# dataframe转成array
X_train_time = Xtrain_time.values
X_train_price = x_train_price_normalise.values
X_train = np.append(X_train_time,X_train_price,axis=1)
Y_train = y_train_price_normalise.values
X_test_time = Xtest_time.values
X_test_price = x_test_price_normalise.values
X_test = np.append(X_test_time,X_test_price,axis=1)
Y_test = y_test_price.values

x_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(Y_train).astype(np.float32)
x_test = np.asarray(X_test).astype(np.float32)

# reshape input to 3D [samples, time_steps, features]
train_x = np.reshape(x_train,(729,24,X_train.shape[1]))
train_y = np.reshape(y_train,(729,24,Y_train.shape[1]))
test_x = np.reshape(x_test,(365,24,X_test.shape[1]))

#%%
#build and fit the LSTM model
#model = Sequential()
#model.add(LSTM(50,input_shape=(24,22)))
#model.add(Dense(24))
#model.compile(loss='mean_squared_error',optimizer='adam')
#model.fit(train_x,train_y,epochs=10,verbose=2)
#%%
# predict in test set and train set
model = tf.keras.models.load_model('elePriceForecast_net.h5')
test_prediction = model.predict(test_x)
y_prediction = test_prediction*189.9+1.57
y_prediction_1row = y_prediction.reshape(8760,1)
pre_datetime = pd.date_range('2019/1/1',periods = 8760, freq='H')
pre_withTime = DataFrame(y_prediction_1row, index=pre_datetime)

train_prediction = model.predict(train_x)
y_prediction_train = train_prediction*189.9+1.57
y_prediction_train_1row = y_prediction_train.reshape(17496,1)
pretrain_datetime = pd.date_range('2017/1/2',periods = 17496, freq='H')
pretrain_withTime = DataFrame(y_prediction_train_1row, index=pretrain_datetime)
# process data to excel
pre_withTime.to_excel('preElePrice_test2019.xlsx')
y_test_price.to_excel('actElePrice_test2019.xlsx')
pretrain_withTime.to_excel('preElePrice_train2017to2018.xlsx')
y_train_price.to_excel('actElePrice_train2017to2018.xlsx')
# pick results in test set : first week, last week, medium week
pre_first = pre_withTime.loc['2019-01-07':'2019-01-13'].values.T
act_first = y_test_price.loc['2019-01-07':'2019-01-13'].values.T
pre_last = pre_withTime.loc['2019-12-23':'2019-12-29'].values.T
act_last = y_test_price.loc['2019-12-23':'2019-12-29'].values.T
pre_mid = pre_withTime.loc['2019-07-01':'2019-07-07'].values.T
act_mid = y_test_price.loc['2019-07-01':'2019-07-07'].values.T

from python_plot.Ploting.fast_plot_Func import *
from python_plot.Ploting.uncertainty_plot_Func import *
def eg_series_uncertainty_plot():
    x = np.arange(0, 168, 1.0)
    ax = series(x=np.concatenate([x, [168.]]),
                y=np.concatenate([act_mid.flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="N2EX day-ahead auction electricity price (GB/MWh)",
                linestyle="-",
                color="tab:red",
                label="Actual Price",
                linewidth="2",
                x_ticks=([0, 24, 48, 72, 96, 120, 144], ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.","Sat.","Sun."]),
                x_lim=(-1.0, 169.0),
                #y_lim=(-10, 150),
                legend_loc='upper center',
                legend_ncol=1)
    ax = series(x=x,
                y=pre_mid.flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:blue",
                linewidth="1.5",
                label="Forecasted Price",
                legend_loc='upper center',
                legend_ncol=2)


if __name__ == "__main__":
    eg_series_uncertainty_plot()

