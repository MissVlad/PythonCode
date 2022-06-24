import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense,Dropout
from matplotlib import pyplot as plt
from scipy.io import loadmat
#%% 这一块看作知识展示，和此模型没有直接关系
#eleMarketData = pd.read_excel("EleMarketData20170101to20211025.xlsx")
#ele_datetime = pd.date_range('2017/1/1', periods=42216, freq='H')
#data = eleMarketData.values.tolist()
#MarketData = DataFrame(data, index=ele_datetime)
#MarketData.columns = ['TF','Year','Month','Day','DayofWeek','Holiday','BST','Covid19','Price','Buy','Sell']
# 数据集划分train，validation和test集;
# input - time information on the forecasted day (month/dayofweek/holi/bst/covid19) and price from **pre 7 days**
# train&validation - 2017.1.1 to 2021.4.30; test - 2021.5.1 to 2021.10.24
# 数据的train、validation、test集划分，以及标准化在MATLAB完成
data = loadmat('XY_data.mat')
Xtrain = data['xtrain']
Xvali = data['xvali']
Xtest = data['xtest']
Ytrain = data['ytrain']
Yvali = data['yvali']
Ytest = data['ytest']

# reshape input to 3D [samples, time_steps, features]
XTrain = np.reshape(Xtrain,(1200,24,14))
YTrain = np.reshape(Ytrain,(1200,24,1))
XVali = np.reshape(Xvali,(300,24,14))
YVali = np.reshape(Yvali,(300,24,1))
XTest = np.reshape(Xtest,(170,24,14))
YTest = np.reshape(Ytest,(170,24,1))
#%% model structure
#model = Sequential()
#model.add(Bidirectional(LSTM(100,input_shape=(24,14),return_sequences=True)))
#model.add(Dropout(0.5))
#model.add(Bidirectional(LSTM(100,return_sequences=True)))
#model.add(Dropout(0.5))
#model.add(Bidirectional(LSTM(100,return_sequences=True)))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #loss=tf.keras.losses.MSE,
              #metrics=['mae'])
#model.fit(x=XTrain,
          #y=YTrain,
          #verbose=2,
          #epochs=10,
          #validation_data=(XVali,YVali))
#%% model results in test set
model = tf.keras.models.load_model('elePriceForecast_net_useCovid19.h5')
test_prediction = model.predict(XTest)
y_prediction = test_prediction*1538.42-38.8
y_prediction_1row = y_prediction.reshape(4080,1)
pre_datetime = pd.date_range('2021/5/8',periods=4080,freq='H')
pretest_withTime = DataFrame(y_prediction_1row,index=pre_datetime)
acttest_withTime = DataFrame(Ytest,index=pre_datetime)
pretest_withTime.to_excel('preElePrice_testModel1.xlsx')
acttest_withTime.to_excel('actElePrice_testModel1.xlsx')

# pick results in test set : first week, last week, medium week
pre_first = pretest_withTime.loc['2021-05-10':'2021-05-16'].values.T
act_first = acttest_withTime.loc['2021-05-10':'2021-05-16'].values.T
pre_last = pretest_withTime.loc['2021-10-18':'2021-10-24'].values.T
act_last = acttest_withTime.loc['2021-10-18':'2021-10-24'].values.T
pre_mid = pretest_withTime.loc['2021-08-02':'2021-08-08'].values.T
act_mid = acttest_withTime.loc['2021-08-02':'2021-08-08'].values.T
#%%
from python_plot.Ploting.fast_plot_Func import *
from python_plot.Ploting.uncertainty_plot_Func import *
def eg_series_uncertainty_plot():
    x = np.arange(0, 168, 1.0)
    ax = series(x=np.concatenate([x, [168.]]),
                y=np.concatenate([act_last.flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="N2EX day-ahead auction electricity price (GB/MWh)",
                linestyle="-",
                color="tab:red",
                label="Actual Price",
                linewidth="2",
                x_ticks=([0, 24, 48, 72, 96, 120, 144], ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.","Sat.","Sun."]),
                x_lim=(-1.0, 169.0),
                #y_lim=(20, 140),
                legend_loc='upper center',
                legend_ncol=1)
    ax = series(x=x,
                y=pre_last.flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:blue",
                linewidth="1.5",
                label="Forecasted Price",
                legend_loc='upper center',
                legend_ncol=2)


if __name__ == "__main__":
    eg_series_uncertainty_plot()