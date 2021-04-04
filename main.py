# Libraries
import os
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy import signal as sg

# Setup
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

############# Data #############

### Data adquisition

# Read data from textfile, drop first column with indices, and enable datetime column
def read_file(textfile):
    df = pd.read_csv(textfile, sep="\t")
    df.drop(df.columns[0], axis=1, inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    print('Overview of the dataset')
    print(df)
    return df

textfile = "dataset_aquaponics_03232021_03312021.txt"
df = read_file(textfile)

# Organize data in columns
df_grouped = pd.pivot_table(df, index = 'DateTime', columns = 'Label', values = 'Value')
print(df_grouped.head)

# Data visualization
sample_sensor = 116

df_grouped[sample_sensor].plot(x = 'DateTime', y = 'Value', kind = 'line')
plt.xticks(rotation=20)
plt.show()

# Clean data
# Delete constant columns
drop_signal = [12, 13, 14, 15, 22, 23, 24, 25, 33, 35, 42, 43, 44, 46, 48, 49, 56, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 97, 98, 99, 100, 101, 104, 105, 106, 111, 112, 113, 114]
df_grouped = df_grouped.drop(columns = drop_signal)
print(df_grouped.first)

# Catch outliners via z-score
z_scores = st.zscore(df_grouped, axis=1)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3) # HOW TO MODIFY THESE ENTRIES?
df_grouped_filter = df_grouped[filtered_entries]
print(df_grouped_filter.first)

# Data pre-processing


# Training and validation dataset

# Data visualization

############# Model architecture #############

# Linear model
def f_linear_model():
    linear_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=64),
                    tf.keras.layers.Dense(units=1)])
    return linear_model

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