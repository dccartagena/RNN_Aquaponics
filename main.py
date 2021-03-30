# Libraries
import os
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Setup
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

############# Data #############

### Data adquisition

# TODO: READ DATA FROM .txt
# TODO: SPLIT SENSORS IN DIFFERENT TABLES
# TODO: DATA VISUALIZATION
# TODO: DATA CLEANING AND PRE-PROCESSING

# Data pre-processing

# Training and validation dataset

# Data visualization

############# Model architecture #############

# Linear model
linear_model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=64),
                tf.keras.layers.Dense(units=1)])

# MLP model
# mlp_model = tf.keras.Sequential([
#                 tf.keras.layers.Dense(units=64, activation='relu'),
#                 tf.keras.layers.Dense(units=1)])

# RNN - Vanilla
# rnn_model = 2


# RNN - LSTM
# lstm_model = 2


# RNN - GRU
# gru_model = 2


############# Deployment #############

# Training
def compile_and_fit(model, train_data, train_label):
    
    model.summary()

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(train_data, 
                        train_label, 
                        batch_size = 128, 
                        epochs = 2,
                        verbose = 2)

    return history

# Prediction