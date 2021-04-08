# Libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy import signal as sg
import seaborn as sns
import tensorflow as tf

# Setup
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

np.random.seed(10)

############# Data #############

### Data adquisition

# Read data from textfile, drop first column with indices, and enable datetime column
def read_file(textfile):
    df = pd.read_csv(textfile, sep="\t")
    df.drop(df.columns[0], axis=1, inplace=True)
    print('Overview of the dataset')
    print(df)
    return df

textfile = "dataset_aquaponics_03232021_03312021.txt"
df = read_file(textfile)

# Organize data in columns
df_grouped = pd.pivot_table(df, index = 'DateTime', columns = 'Label', values = 'Value')
df_grouped.index = pd.to_datetime(df_grouped.index)

start_date = pd.to_datetime('2021-03-23 00:00:00')
end_date = pd.to_datetime('2021-03-25 23:59:59')

df_grouped = df_grouped.loc[start_date:end_date]
print(df_grouped.head)

### Data visualization
sample_sensor = 93

# df_grouped[sample_sensor].plot(x = 'DateTime', y = 'Value', kind = 'line')
# plt.xticks(rotation=20)
# plt.show()

### Data pre-processing

# Delete constant or corrupted-data columns
drop_signal = [12, 13, 14, 15, 22, 23, 24, 25, 32, 33, 35, 42, 43, 44, 46, 48, 49, 56, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 93, 97, 98, 99, 100, 101, 104, 105, 106, 111, 112, 113, 114]
df_grouped  = df_grouped.drop(columns = drop_signal)
print(df_grouped.first)

# Correlation analysis
corr_mat = np.abs(df_grouped.corr(method='pearson'))
# sns.heatmap(corr_mat)
# plt.show()

corr_thl = 0.25
corr_label = (corr_mat[4] > corr_thl) | (corr_mat[5] > corr_thl) | (corr_mat[8] > corr_thl) | (corr_mat[9] > corr_thl) | (corr_mat[18] > corr_thl)
df_grouped = df_grouped.loc[:, corr_label]
print(df_grouped.first)

# Describe dataset
print(df_grouped.describe().transpose())

# Split dataset
total_data = len(df_grouped)

train_perc, val_perc, test_perc = 0.7, 0.2, 0.1

train_df    = df_grouped[0:int(total_data*train_perc)]
val_df      = df_grouped[int(total_data*train_perc):int(total_data*(train_perc + val_perc))]
test_df     = df_grouped[int(total_data*(train_perc + val_perc)):total_data]

# Mean-STD normalization
norm_mean   = train_df.mean()
norm_std    = train_df.std()

norm_train_df   = (train_df - norm_mean) / norm_std
norm_val_df     = (val_df - norm_mean) / norm_std
norm_test_df    = (test_df - norm_mean) / norm_std

# Show tails with box plot
melt_train_df = norm_train_df.melt(var_name = 'Label', value_name = 'Normalized value')
# plt.figure()
# ax = sns.violinplot(x = 'Label', y = 'Normalized value', data = melt_train_df)
# plt.show()

### Window generation
n_hours_in  = 2
n_hours_out = 1

input_width     = 65 * n_hours_in # 65 entries ~ 1 hour - There is a variable sampling frecuency
label_width     = 65 * n_hours_out # Predict 1 hour
offset_width    = 1 # 1 as default

target_label = [4, 5, 8, 9, 18]
# Our target variables are:
# 4     = pH sump B
# 5     = pH sump A
# 8     = % oxygen B
# 9     = % oxygen A
# 18    = C02
# 93*   = Weight cell 1
# 99*   = Nitrate B 
# 100*  = Ammonia B
# 103*  = Nitrate A 
# 104*  = Ammonia A
# * sensors are not online yet

# Create input and label dataframe
def f_window_gen(df, target_label, input_width, label_width, offset_width):
    
    time_range = df.index
    data = []
    label = []

    for i in range(len(time_range) - (input_width + offset_width + label_width)):
        m_data = df.loc[time_range[i]:time_range[input_width + i]].to_numpy()
        data.append(m_data.flatten('C'))

        m_label = df[target_label].loc[time_range[input_width + offset_width + i: input_width + offset_width + label_width + i]].to_numpy()
        label.append(m_label.flatten('C'))

    return data, label

train_data, train_label = f_window_gen(norm_train_df, target_label, input_width, label_width, offset_width)
val_data, val_label = f_window_gen(norm_val_df, target_label, input_width, label_width, offset_width)
test_data, test_label = f_window_gen(norm_test_df, target_label, input_width, label_width, offset_width)

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