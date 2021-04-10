# Libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy import signal as sg
import seaborn as sns
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Setup
mpl.rcParams['figure.figsize'] = (14, 8)
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
end_date = pd.to_datetime('2021-03-26 23:59:59')

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

label_target = {4: 'pH sump B', 5: 'pH sump A', 8: '% oxygen B', 9: '% oxygen A', 18: 'C02'}
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
def f_window_gen(df, label_target, input_width, label_width, offset_width):
    
    time_range = df.index
    data = []
    label = []

    data_time = []
    label_time = []

    for i in range(len(time_range) - (input_width + offset_width + label_width)):
        m_data = df.loc[time_range[i]:time_range[input_width + i]].to_numpy()
        data.append(m_data.flatten('C'))
        data_time.append(time_range[i:input_width + i])

        m_label = df[label_target].loc[time_range[input_width + offset_width + i: input_width + offset_width + label_width + i]].to_numpy()
        label.append(m_label.flatten('C'))
        label_time.append(time_range[input_width + offset_width + i: input_width + offset_width + label_width + i])

    return np.array(data), np.array(label), data_time, label_time


train_data, train_label, train_data_time, train_label_time = f_window_gen(norm_train_df, label_target, input_width, label_width, offset_width)
val_data, val_label, val_data_time, val_label_time = f_window_gen(norm_val_df, label_target, input_width, label_width, offset_width)
test_data, test_label, test_data_time, test_label_time = f_window_gen(norm_test_df, label_target, input_width, label_width, offset_width)

### Plotting - Normalized values
def plot_results(label_target, label_time, label, results):
    n_label = np.int(len(label_target))
    n_time = np.int(label_time.shape[0])

    m_label = label.reshape(n_time, n_label)
    m_results = results.reshape(n_time, n_label)

    sensor_tag = list(label_target.values())

    fig, ax = plt.subplots(n_label, sharex = 'all')

    for i in range(n_label):
        ax[i].plot(label_time, m_label[:, i], color = 'blue', label = 'Real')
        ax[i].plot(label_time, m_results[:, i], color = 'orange', label = 'Prediction')
        ax[i].set_ylabel(sensor_tag[i])

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'center right')
    plt.xticks(rotation=20)
    plt.xlabel('Day - Time')
    plt.show()

    pass

# plot_entry = -1
# results = np.ones(test_label[plot_entry].shape)
# plot_results(label_target, test_label_time[plot_entry], test_label[plot_entry], results)

############# Model architecture #############

# Linear model
def f_linear_model(label_target, label_width):
    linear_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=1000),
                    tf.keras.layers.Dense(units=700),
                    tf.keras.layers.Dense(units=500),
                    tf.keras.layers.Dense(units=400),
                    tf.keras.layers.Dense(units = len(label_target) * label_width)])
    return linear_model

linear_model = f_linear_model(label_target, label_width)

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
def compile_and_fit(model, train_data, train_label, test_data, test_label, max_epochs):
    
    train_data  = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
    test_data   = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
    
    train_dataset   = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    test_dataset    = tf.data.Dataset.from_tensor_slices((test_data, test_label))

    # Add callbacks

    # TODO: MULTIPLE OUTPUTS
    # TODO: PREDICT ONE HOUR

    # Model compile
    model.compile(loss      = tf.losses.MeanSquaredError(),
                  optimizer = tf.optimizers.Adam(),
                  metrics   = [tf.metrics.MeanAbsoluteError()])

    # Model fit
    history = model.fit(train_data,
                        train_label,
                        batch_size = 65,
                        epochs  = max_epochs,
                        verbose = 2,
                        shuffle =False)

    model.evaluate(test_data, test_label)

    test_prediction = model.predict(test_data[0][:][:])
    return history, test_prediction

max_epochs = 50
history_linear, test_prediction = compile_and_fit(linear_model, train_data, train_label, test_data, test_label, max_epochs)

results = test_prediction.reshape(label_width, len(label_target))

plot_entry = 0
plot_results(label_target, test_label_time[plot_entry], test_label[plot_entry], results)