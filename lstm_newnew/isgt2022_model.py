import numpy as np
import tensorflow as tf
from File_Management.path_and_file_management_Func import *
from project_utils import *
from Ploting.fast_plot_Func import *
import pandas as pd
import time
from File_Management.load_save_Func import *
import json

BATCH_SIZE = 5000


class ModelBaseClass:
    __slots__ = ("price_x_shape",
                 "price_y_shape",
                 "buy_x_shape",
                 "buy_y_shape",
                 "sell_x_shape",
                 "sell_y_shape",
                 "save_folder_path")

    def __init__(self,
                 price_x_shape,
                 price_y_shape,
                 buy_x_shape,
                 buy_y_shape,
                 sell_x_shape,
                 sell_y_shape,
                 save_folder_path):
        self.price_x_shape = price_x_shape
        self.price_y_shape = price_y_shape
        self.buy_x_shape = buy_x_shape
        self.buy_y_shape = buy_y_shape
        self.sell_x_shape = sell_x_shape
        self.sell_y_shape = sell_y_shape

        try_to_find_folder_path_otherwise_make_one(save_folder_path)
        self.save_folder_path = save_folder_path

    def _get_model(self):
        # Price
        price_layer_input = tf.keras.Input(
            shape=self.price_x_shape,
            name="price_layer_input"
        )

        price_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001),),
            name="price_layer_output"
        )(price_layer_input)

        price_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="price_layer_output_dropout"
        )(price_layer_output)

        price_layer_output_dense = tf.keras.layers.Dense(
            self.price_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="price_layer_output_dense"
        )(price_layer_output_dropout)

        price_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="price_layer_output_dense_dropout"
        )(price_layer_output_dense)

        # Buy
        buy_layer_input = tf.keras.Input(
            shape=self.buy_x_shape,
            name="buy_layer_input"
        )
        buy_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001),),
            name="buy_layer_output"
        )(buy_layer_input)
        buy_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="buy_layer_output_dropout"
        )(buy_layer_output)
        buy_layer_output_dense = tf.keras.layers.Dense(
            self.buy_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="buy_layer_output_dense"
        )(buy_layer_output_dropout)
        buy_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="buy_layer_output_dense_dropout"
        )(buy_layer_output_dense)

        # Sell
        sell_layer_input = tf.keras.Input(
            shape=self.sell_x_shape,
            name="sell_layer_input"
        )
        sell_layer_output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(12, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001),),
            name="sell_layer_output"
        )(sell_layer_input)
        sell_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="sell_layer_output_dropout"
        )(sell_layer_output)
        sell_layer_output_dense = tf.keras.layers.Dense(
            self.sell_y_shape[1],
            name="sell_layer_output_dense",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(sell_layer_output_dropout)
        sell_layer_output_dense = tf.keras.layers.Dropout(
            0.5,
            name="sell_layer_output_dense_dropout"
        )(sell_layer_output_dense)

        # All in one
        x = tf.keras.layers.concatenate(
            [price_layer_output_dense,
             buy_layer_output_dense,
             sell_layer_output_dense],
            name="concatenate",
            axis=2
        )

        # Forecast
        correlated = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(24, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001),),
            name="correlated"
        )(x)
        correlated_layer_output_dropout = tf.keras.layers.Dropout(
            0.5,
            name="correlated_layer_output_dropout"
        )(correlated)
        correlated_forecasting = tf.keras.layers.Dense(
            self.price_y_shape[1],
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="correlated_forecasting"
        )(correlated_layer_output_dropout)

        model = tf.keras.Model(
            inputs=[price_layer_input, buy_layer_input, sell_layer_input],
            outputs=correlated_forecasting,
        )

        return model

    def train(self,
              price_training_data_x,
              price_training_data_y,
              buy_training_data_x,
              buy_training_data_y,  # unused
              sell_training_data_x,
              sell_training_data_y,  # unused
              price_validation_data,
              buy_validation_data,
              sell_validation_data,
              epoch):
        model = self._get_model()
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss=['mse'],
                      metrics=['mae', 'mse'])
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=int(epoch * 0.3))
        save_folder_path = self.save_folder_path
        history = dict()

        class SaveCallback(tf.keras.callbacks.Callback):
            start_time = time.time()

            def on_epoch_end(self, _epoch, logs=None):
                freq = 50

                if _epoch % int(epoch * 0.025) == 0:
                    model.save_weights(save_folder_path / fr'model_epoch_{_epoch}.h5')

                if _epoch % freq == 0:
                    print(f"Epoch {_epoch}/{epoch}\n"
                          f"loss = {logs.get('loss'):.4f}, "
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
                    print(f"Average run time = {delta_time / freq:.4f} sec/epoch, ",
                          f"waiting for another maximum {(epoch - _epoch) / freq * delta_time / 3600:.4f} hours",
                          flush=True)

        model.fit(
            x=[price_training_data_x, buy_training_data_x, sell_training_data_x],
            y=price_training_data_y,
            verbose=0,
            batch_size=BATCH_SIZE,
            validation_batch_size=BATCH_SIZE // 2,
            epochs=epoch,
            validation_data=(
                [price_validation_data[0], buy_validation_data[0], sell_validation_data[0]],
                price_validation_data[1]
            ),
            callbacks=[SaveCallback(), stop_early]
        )

        model.save_weights(save_folder_path / "final.h5")
        save_pkl_file(save_folder_path / "history.pkl", history)
        with open(save_folder_path / "history.json", 'w') as fp:
            json.dump(history, fp)

    def predict(self,
                price_data_x,
                buy_data_x,
                sell_data_x):
        model = self._get_model()
        model.load_weights(self.save_folder_path / 'final.h5')
        y_pred = model.predict([price_data_x, buy_data_x, sell_data_x])

        return y_pred
