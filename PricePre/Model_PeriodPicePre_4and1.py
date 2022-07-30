import numpy as np
import tensorflow as tf
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from project_utils import *
from Code.Ploting.fast_plot_Func import *
import pandas as pd
import time
import json

class PricePeriodPreClass:
    __slots__ = ("price_period1_x_shape",
                 "price_period1_y_shape",
                 "price_period2_x_shape",
                 "price_period2_y_shape",
                 "price_period3_x_shape",
                 "price_period3_y_shape",
                 "price_period4_x_shape",
                 "price_period4_y_shape",
                 "price_period5_x_shape",
                 "price_period5_y_shape",
                 "price_dayahead_x_shape",
                 "price_dayahead_y_shape",
                 "save_folder_path") # 不懂

    def __init__(self,
                 price_period1_x_shape,
                 price_period1_y_shape,
                 price_period2_x_shape,
                 price_period2_y_shape,
                 price_period3_x_shape,
                 price_period3_y_shape,
                 price_period4_x_shape,
                 price_period4_y_shape,
                 price_period5_x_shape,
                 price_period5_y_shape,
                 price_dayahead_x_shape,
                 price_dayahead_y_shape,
                 save_folder_path):
        self.price_period1_x_shape = price_period1_x_shape
        self.price_period1_y_shape = price_period1_y_shape
        self.price_period2_x_shape = price_period2_x_shape
        self.price_period2_y_shape = price_period2_y_shape
        self.price_period3_x_shape = price_period3_x_shape
        self.price_period3_y_shape = price_period3_y_shape
        self.price_period4_x_shape = price_period4_x_shape
        self.price_period4_y_shape = price_period4_y_shape
        self.price_period5_x_shape = price_period5_x_shape
        self.price_period5_y_shape = price_period5_y_shape
        self.price_dayahead_x_shape_y_shape = price_dayahead_x_shape
        self.price_dayahead_y_shape_y_shape = price_dayahead_y_shape

        try_to_find_folder_path_otherwise_make_one(save_folder_path) # 不懂
        self.save_folder_path = save_folder_path # 不懂

    def _get_model(self):
        # period 1
        period1_layer_input = tf.keras.Input(
            shape = self.price_period1_x_shape,
            name = "period1_layer_input"
        )

        period1_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
            name="period1_layer_output"
        )(period1_layer_input)

        period1_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name = "period1_layer_output_dropout"
        )(period1_layer_output)

        period1_layer_output_dense = tf.keras.layers.Dense(
            self.price_period1_y_shape[1],
            kernel_regularizer = tf.keras.regularizers.l2(0.001),
            name = "period1_layer_output_dense"
        )(period1_layer_output_dropout)

        period1_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name = "period1_layer_output_dense_dropout"
        )(period1_layer_output_dense)

        # period 2
        period2_layer_input = tf.keras.Input(
            shape=self.price_period2_x_shape,
            name="period2_layer_input"
        )

        period2_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
            name="period2_layer_output"
        )(period2_layer_input)

        period2_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="period2_layer_output_dropout"
        )(period2_layer_output)

        period2_layer_output_dense = tf.keras.layers.Dense(
            self.price_period2_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="period2_layer_output_dense"
        )(period2_layer_output_dropout)

        period2_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="period2_layer_output_dense_dropout"
        )(period2_layer_output_dense)

        # period 3
        period3_layer_input = tf.keras.Input(
            shape=self.price_period3_x_shape,
            name="period3_layer_input"
        )

        period3_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
            name="period3_layer_output"
        )(period3_layer_input)

        period3_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="period3_layer_output_dropout"
        )(period3_layer_output)

        period3_layer_output_dense = tf.keras.layers.Dense(
            self.price_period3_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="period3_layer_output_dense"
        )(period3_layer_output_dropout)

        period3_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="period3_layer_output_dense_dropout"
        )(period3_layer_output_dense)

        # period 4
        period4_layer_input = tf.keras.Input(
            shape=self.price_period4_x_shape,
            name="period4_layer_input"
        )

        period4_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
            name="period4_layer_output"
        )(period4_layer_input)

        period4_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="period4_layer_output_dropout"
        )(period4_layer_output)

        period4_layer_output_dense = tf.keras.layers.Dense(
            self.price_period4_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="period4_layer_output_dense"
        )(period4_layer_output_dropout)

        period4_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="period4_layer_output_dense_dropout"
        )(period4_layer_output_dense)

        # period 5
        period5_layer_input = tf.keras.Input(
            shape=self.price_period5_x_shape,
            name="period5_layer_input"
        )

        period5_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
            name="period5_layer_output"
        )(period5_layer_input)

        period5_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="period5_layer_output_dropout"
        )(period5_layer_output)

        period5_layer_output_dense = tf.keras.layers.Dense(
            self.price_period5_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="period5_layer_output_dense"
        )(period5_layer_output_dropout)

        period5_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="period5_layer_output_dense_dropout"
        )(period5_layer_output_dense)

        # day-ahead
        day_layer_input = tf.keras.Input(
            shape=self.price_dayahead_x_shape_x_shape,
            name="day_layer_input"
        )

        day_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001), ),
            name="day_layer_output"
        )(day_layer_input)

        day_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="day_layer_output_dropout"
        )(day_layer_output)

        day_layer_output_dense = tf.keras.layers.Dense(
            self.price_dayahead_y_shape_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="day_layer_output_dense"
        )(day_layer_output_dropout)

        day_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="day_layer_output_dense_dropout"
        )(day_layer_output_dense)

        # all in one
        x = tf.keras.layers.concatenate(
            [period1_layer_output_dense,
             period2_layer_output_dense,
             period3_layer_output_dense,
             period4_layer_output_dense,
             period5_layer_output_dense,
             day_layer_output_dense],
            name = "concatenate",
            axis = 2
        )

        # forecast
        correlated = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(24,return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001),),
            name="correlated"
        )(x)
        correlated_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="correlated_layer_output_dropout"
        )(correlated)
        correlated_forecasting = tf.keras.layers.Dense(
            self.price_dayahead_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="correlated_forecasting"
        )(correlated_layer_output_dropout)

        model = tf.keras.Model(
            inputs=[period1_layer_input,period2_layer_input,period3_layer_input,period4_layer_input,period5_layer_input,day_layer_input],
            outputs=correlated_forecasting
        )

        return model

    def train(self,
              period1_training_x,
              period2_training_x,
              period3_training_x,
              period4_training_x,
              period5_training_x,
              dayahead_training_x,
              dayahead_training_y,
              period1_validation,
              period2_validation,
              period3_validation,
              period4_validation,
              period5_validation,
              dayahead_validation,
              epoch):
        model = self._get_model()
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss=['mse'],
                      metrics=['mae','mse'])
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=int(epoch * 0.3))
        save_folder_path = self.save_folder_path
        history = dict()

        class SaveCallback(tf.keras.callbacks.Callback):
            start_time = time.time()

            def on_epoch_end(self,_epoch, logs=None):
                freq = 50

                if _epoch % freq == 0:
                    print(f"Epoch {_epoch}/{epoch}\n"
                          f"loss = {logs.get('loss'):.4f},"
                          f"mae = {logs.get('mae'):.4f}, "
                          f"mse = {logs.get('mse'):.4f} |",
                          f"val_loss = {logs.get('val_loss'):.4f}, "
                          f"val_mae = {logs.get('val_mae'):.4f}, "
                          f"val_mse = {logs.get('val_mse'):.4f}",
                          flush=True)
                    history[_epoch] = {
                        'loss': logs.get('loss'),
                        'mae': logs.get('mae'),
                        'mse': logs.get('mse'),
                        'val_loss': logs.get('val_loss'),
                        'val_mae': logs.get('val_mae'),
                        'val_mse': logs.get('val_mse')
                    }
                    delta_time = time.time() - self.start_time
                    self.start_time = time.time()
                    print(f"Average run time = {delta_time / freq:.4f} sec/epoch,",
                          f"waiting for another maximum {(epoch - _epoch) / freq * delta_time / 3600:.4f} hours",
                          flush=True)

                    model.fit(
                        x=[period1_training_x, period2_training_x, period3_training_x, period4_training_x, period5_training_x, dayahead_training_x],
                        y=dayahead_training_y,
                        verbose=0,
                        batch_size=BATCH_SIZE,
                        validation_batch_size=BATCH_SIZE // 2,
                        epochs=epoch,
                        validation_data=(
                            [period1_validation[0],period2_validation[0],period3_validation[0],period4_validation[0],period5_validation[0],dayahead_validation[0]],
                            dayahead_validation[1]
                        ),
                        callback=[SaveCallback(), stop_early]
                    )

                    model.save_weights(save_folder_path / "final.h5")
                    save_pkl_file(save_folder_path / "history.pkl",history)
                    with open(save_folder_path / "history.json",'w') as fp:
                        json.dump(history, fp)

                def predict(self,
                            period1_x,
                            period2_x,
                            period3_x,
                            period4_x,
                            period5_x,
                            dayahead_x):
                    model = self._get_model()
                    model.load_weights(self.save_folder_path / 'final.h5')
                    y_pred = model.predict([period1_x,period2_x,period3_x,period4_x,period5_x,dayahead_x])

                    return y_pred






























