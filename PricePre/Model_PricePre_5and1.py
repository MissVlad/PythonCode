# %%
import pandas as pd
from pandas import DataFrame
import openpyxl
import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.io import loadmat
from Code.project_utils import project_path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 这个模型使用1个concatenate连接
# 这个模型预测出的步长为100，即五个区间预测数值，和一天预测数值

data = loadmat(project_path / 'data/processed_data/N2EXMarketdata/data_XYTrainValiTest_5and1.mat')
xtrain_p1 = data['xtrain_period1']
ytrain_p1 = data['ytrain_period1']
xvali_p1 = data['xvali_period1']
yvali_p1 = data['yvali_period1']
xtest_p1 = data['xtest_period1']

xtrain_p2 = data['xtrain_period2']
ytrain_p2 = data['ytrain_period2']
xvali_p2 = data['xvali_period2']
yvali_p2 = data['yvali_period2']
xtest_p2 = data['xtest_period2']

xtrain_p3 = data['xtrain_period3']
ytrain_p3 = data['ytrain_period3']
xvali_p3 = data['xvali_period3']
yvali_p3 = data['yvali_period3']
xtest_p3 = data['xtest_period3']

xtrain_p4 = data['xtrain_period4']
ytrain_p4 = data['ytrain_period4']
xvali_p4 = data['xvali_period4']
yvali_p4 = data['yvali_period4']
xtest_p4 = data['xtest_period4']

xtrain_p5 = data['xtrain_period5']
ytrain_p5 = data['ytrain_period5']
xvali_p5 = data['xvali_period5']
yvali_p5 = data['yvali_period5']
xtest_p5 = data['xtest_period5']

xtrain_dayahead = data['xtrain_dayahead']
ytrain_dayahead = data['ytrain_dayahead']
xvali_dayahead = data['xvali_dayahead']
yvali_dayahead = data['yvali_dayahead']
xtest_dayahead = data['xtest_dayahead']
ytest_dayahead = data['ytest_dayahead']

# reshape input to 3D [samples, time_steps, features]
Xtrain_p1 = np.reshape(xtrain_p1,(630,10,13))
Ytrain_p1 = np.reshape(ytrain_p1,(630,10,1))
Xvali_p1 = np.reshape(xvali_p1,(70,10,13))
Yvali_p1 = np.reshape(yvali_p1,(70,10,1))
Xtest_p1 = np.reshape(xtest_p1,(358,10,13))

Xtrain_p2 = np.reshape(xtrain_p2,(630,9,13))
Ytrain_p2 = np.reshape(ytrain_p2,(630,9,1))
Xvali_p2 = np.reshape(xvali_p2,(70,9,13))
Yvali_p2 = np.reshape(yvali_p2,(70,9,1))
Xtest_p2 = np.reshape(xtest_p2,(358,9,13))

Xtrain_p3 = np.reshape(xtrain_p3,(630,13,13))
Ytrain_p3 = np.reshape(ytrain_p3,(630,13,1))
Xvali_p3 = np.reshape(xvali_p3,(70,13,13))
Yvali_p3 = np.reshape(yvali_p3,(70,13,1))
Xtest_p3 = np.reshape(xtest_p3,(358,13,13))

Xtrain_p4 = np.reshape(xtrain_p4,(630,7,13))
Ytrain_p4 = np.reshape(ytrain_p4,(630,7,1))
Xvali_p4 = np.reshape(xvali_p4,(70,7,13))
Yvali_p4 = np.reshape(yvali_p4,(70,7,1))
Xtest_p4 = np.reshape(xtest_p4,(358,7,13))

Xtrain_p5 = np.reshape(xtrain_p5,(630,13,13))
Ytrain_p5 = np.reshape(ytrain_p5,(630,13,1))
Xvali_p5 = np.reshape(xvali_p5,(70,13,13))
Yvali_p5 = np.reshape(yvali_p5,(70,13,1))
Xtest_p5 = np.reshape(xtest_p5,(358,13,13))

Xtrain_dayahead = np.reshape(xtrain_dayahead, (630, 48, 13))
Ytrain_dayahead = np.reshape(ytrain_dayahead, (630, 48, 1))
Xvali_dayahead = np.reshape(xvali_dayahead, (70, 48, 13))
Yvali_dayahead = np.reshape(yvali_dayahead, (70, 48, 1))
Xtest_dayahead = np.reshape(xtest_dayahead, (358, 48, 13))
Ytest_dayahead = np.reshape(ytest_dayahead, (358, 48, 1))

# %% model structure
# period 1
period1_layer_input = tf.keras.Input(
    shape=(10,13),
    name="period1_layer_input")
period1_layer_output = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
    name="period1_layer_output")(period1_layer_input)
period1_layer_output_dropout = tf.keras.layers.Dropout(
    0.5,
    name="period1_layer_output_dropout")(period1_layer_output)
period1_layer_output_dense = tf.keras.layers.Dense(
    1,
    kernel_regularizer = tf.keras.regularizers.l2(0.001),
    name="period1_layer_output_dense")(period1_layer_output_dropout)
period1_layer_output_dense = tf.keras.layers.Dropout(
    0.5,
    name="period1_layer_output_dense_dropout")(period1_layer_output_dense)

# period 2
period2_layer_input = tf.keras.Input(
    shape=(9,13),
    name="period2_layer_input")
period2_layer_output = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
    name="period2_layer_output")(period2_layer_input)
period2_layer_output_dropout = tf.keras.layers.Dropout(
    0.5,
    name="period2_layer_output_dropout")(period2_layer_output)
period2_layer_output_dense = tf.keras.layers.Dense(
    1,
    kernel_regularizer = tf.keras.regularizers.l2(0.001),
    name="period2_layer_output_dense")(period2_layer_output_dropout)
period2_layer_output_dense = tf.keras.layers.Dropout(
    0.5,
    name="period2_layer_output_dense_dropout")(period2_layer_output_dense)

# period 3
period3_layer_input = tf.keras.Input(
    shape=(13,13),
    name="period3_layer_input")
period3_layer_output = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
    name="period3_layer_output")(period3_layer_input)
period3_layer_output_dropout = tf.keras.layers.Dropout(
    0.5,
    name="period3_layer_output_dropout")(period3_layer_output)
period3_layer_output_dense = tf.keras.layers.Dense(
    1,
    kernel_regularizer = tf.keras.regularizers.l2(0.001),
    name="period3_layer_output_dense")(period3_layer_output_dropout)
period3_layer_output_dense = tf.keras.layers.Dropout(
    0.5,
    name="period3_layer_output_dense_dropout")(period3_layer_output_dense)

# period 4
period4_layer_input = tf.keras.Input(
    shape=(7,13),
    name="period4_layer_input")
period4_layer_output = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
    name="period4_layer_output")(period4_layer_input)
period4_layer_output_dropout = tf.keras.layers.Dropout(
    0.5,
    name="period4_layer_output_dropout")(period4_layer_output)
period4_layer_output_dense = tf.keras.layers.Dense(
    1,
    kernel_regularizer = tf.keras.regularizers.l2(0.001),
    name="period4_layer_output_dense")(period4_layer_output_dropout)
period4_layer_output_dense = tf.keras.layers.Dropout(
    0.5,
    name="period4_layer_output_dense_dropout")(period4_layer_output_dense)

# period 5
period5_layer_input = tf.keras.Input(
    shape=(13,13),
    name="period5_layer_input")
period5_layer_output = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
    name="period5_layer_output")(period5_layer_input)
period5_layer_output_dropout = tf.keras.layers.Dropout(
    0.5,
    name="period5_layer_output_dropout")(period5_layer_output)
period5_layer_output_dense = tf.keras.layers.Dense(
    1,
    kernel_regularizer = tf.keras.regularizers.l2(0.001),
    name="period5_layer_output_dense")(period5_layer_output_dropout)
period5_layer_output_dense = tf.keras.layers.Dropout(
    0.5,
    name="period5_layer_output_dense_dropout")(period5_layer_output_dense)

# day ahead
day_layer_input = tf.keras.Input(
    shape=(48,13),
    name="day_layer_input")
day_layer_output = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
    name="day_layer_output")(day_layer_input)
day_layer_output_dropout = tf.keras.layers.Dropout(
    0.5,
    name="day_layer_output_dropout")(day_layer_output)
day_layer_output_dense = tf.keras.layers.Dense(
    1,
    kernel_regularizer = tf.keras.regularizers.l2(0.001),
    name="day_layer_output_dense")(day_layer_output_dropout)
day_layer_output_dense = tf.keras.layers.Dropout(
    0.5,
    name="day_layer_output_dense_dropout")(day_layer_output_dense)

# all in one
x = tf.keras.layers.concatenate(
    [period1_layer_output_dense,
     period2_layer_output_dense,
     period3_layer_output_dense,
     period4_layer_output_dense,
     period5_layer_output_dense,
     day_layer_output_dense],
    name="concatenate",
    axis=1)

# forecast
correlated = tf.keras.layers.Bidirectional(
tf.keras.layers.LSTM(24,return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001),),
    name="correlated")(x)
correlated_layer_output_dropout = tf.keras.layers.Dropout(
    0.5,
    name="correlated_layer_output_dropout")(correlated)
correlated_forecasting = tf.keras.layers.Dense(
    1,
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
    name="correlated_forecasting")(correlated_layer_output_dropout)

model = tf.keras.Model(
    inputs=[period1_layer_input,period2_layer_input,period3_layer_input,period4_layer_input,period5_layer_input,day_layer_input],
    outputs=correlated_forecasting)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss=['mse'],
              metrics=['mae', 'mse'])


Ytrain = np.hstack((Ytrain_p1,Ytrain_p2,Ytrain_p3,Ytrain_p4,Ytrain_p5,Ytrain_dayahead))
Yvali = np.hstack((Yvali_p1,Yvali_p2,Yvali_p3,Yvali_p4,Yvali_p5,Yvali_dayahead))

model.fit(
    x=[Xtrain_p1, Xtrain_p2, Xtrain_p3, Xtrain_p4, Xtrain_p5, Xtrain_dayahead],
    y=Ytrain,
    verbose=2,
    epochs=5000,
    validation_data=([Xvali_p1,Xvali_p2,Xvali_p3,Xvali_p4,Xvali_p5,Xvali_dayahead],
                     Yvali))

model.save('Net_PricePre_5and1_first.h5')

# %% minimum price = 18.9500, maximum price = 191.5500
model = tf.keras.models.load_model(project_path /'Code/PricePre/Net_PricePre_5and1_first.h5')
test_pre = model.predict([Xtest_p1,Xtest_p2,Xtest_p3,Xtest_p4,Xtest_p5,Xtest_dayahead])
test_pre_retureMinMax = test_pre * (191.5500-18.9500) + 18.9500
# %%
test_pre_reshape = test_pre_retureMinMax.reshape(35800,1)
test_pre_dataframe = pd.DataFrame(test_pre_reshape)
test_pre_dataframe.to_excel('data_PricePre2019_5and1_first.xlsx')
# %%
test_pre_reshape2 = test_pre_retureMinMax.reshape(358,100)
test_pre_dataframe2 = pd.DataFrame(test_pre_reshape2)
test_pre_dataframe2.to_excel('data_PricePre2019_5and1_First.xlsx')






