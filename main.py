# Libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    print('Dataset ready')
    return df

textfile = "dataset_aquaponics_03232021_04232021.txt"
df = read_file(textfile)

# Organize data in columns
df_grouped = pd.pivot_table(df, index = 'DateTime', columns = 'Label', values = 'Value')
df_grouped.index = pd.to_datetime(df_grouped.index)

start_date = pd.to_datetime('2021-04-01 00:00:00')
end_date = pd.to_datetime('2021-04-10 23:59:59')

df_grouped = df_grouped.loc[start_date:end_date]
# print(df_grouped.head)

### Data visualization
sample_sensor = 93

# df_grouped[sample_sensor].plot(x = 'DateTime', y = 'Value', kind = 'line')
# plt.xticks(rotation=20)
# plt.show()

### Data pre-processing

# Target data
label_target = {4: 'pH sump B', 
                5: 'pH sump A', 
                8: '% oxygen B', 
                9: '% oxygen A', 
                18: 'C02'}
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

# Delete constant or corrupted-data columns
drop_signal = [12, 13, 14, 15, 22, 23, 24, 25, 32, 33, 35, 42, 43, 44, 46, 48, 49, 56, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 93, 97, 98, 99, 100, 101, 103, 104, 105, 106, 111, 112, 113, 114]
df_grouped  = df_grouped.drop(columns = drop_signal)
# print(df_grouped.first)

# Correlation analysis
corr_mat = np.abs(df_grouped.corr(method='pearson'))
# sns.heatmap(corr_mat)
# plt.show()

corr_thl = 0.50
corr_label = np.zeros(df_grouped.shape[1])

for i in label_target:
    corr_label = corr_label | (corr_mat[i] > corr_thl)

df_grouped = df_grouped.loc[:, corr_label]
# print(df_grouped.first)

# Describe dataset
# print(df_grouped.describe().transpose())

# Split dataset
total_data = len(df_grouped)

train_perc, val_perc, test_perc = 0.7, 0.2, 0.1

train_df    = df_grouped[0:int(total_data*train_perc)]
val_df      = df_grouped[int(total_data*train_perc):int(total_data*(train_perc + val_perc))]
test_df     = df_grouped[int(total_data*(train_perc + val_perc)):total_data]

# # Mean-STD normalization
norm_mean   = train_df.mean()
norm_std    = train_df.std()

norm_train_df   = (train_df - norm_mean) / norm_std
norm_val_df     = (val_df - norm_mean) / norm_std
norm_test_df    = (test_df - norm_mean) / norm_std

# # Min-max normalization
# norm_train_df   = (train_df - train_df.min()) / (train_df.max() - train_df.min())
# norm_val_df     = (val_df - train_df.min()) / (train_df.max() - train_df.min())
# norm_test_df    = (test_df - train_df.min()) / (train_df.max() - train_df.min())

# # TODO: Min-Max percentile normalization


# # Show tails with box plot
# melt_train_df = norm_train_df.melt(var_name = 'Label', value_name = 'Normalized value')
# # plt.figure()
# # ax = sns.violinplot(x = 'Label', y = 'Normalized value', data = melt_train_df)
# # plt.show()

### Window generation
# # Create input and target dataframe
n_hour_data = 65 # 65 entries ~ 1 hour - There is a variable sampling frecuency
n_hours_in  = 1
n_hours_out = 1

input_width     = n_hour_data * n_hours_in 
label_width     = n_hour_data * n_hours_out 
offset_width    = 1 # 1 as default

def f_window_gen(df, label_target, input_width, label_width, offset_width):

    time_range = df.index
    data = []
    target = []

    data_time = []
    target_time = []

    for i in range(len(time_range) - (input_width + offset_width + label_width)):
        range_data = df.loc[time_range[i]:time_range[input_width + i]].values
        data.append(range_data)
        data_time.append(time_range[i:input_width + i])

        range_target = df[label_target].loc[time_range[input_width + offset_width + i: input_width + offset_width + label_width + i]].values
        target.append(range_target.flatten('C'))
        target_time.append(time_range[input_width + offset_width + i: input_width + offset_width + label_width + i])

    data = np.array(data)
    target = np.array(target)

    return data, target, data_time, target_time

train_data, train_target, train_data_time, train_target_time = f_window_gen(norm_train_df, label_target, input_width, label_width, offset_width)
val_data, val_target, val_data_time, val_target_time = f_window_gen(norm_val_df, label_target, input_width, label_width, offset_width)
test_data, test_target, test_data_time, test_target_time = f_window_gen(norm_test_df, label_target, input_width, label_width, offset_width)

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

### Plotting - History
def plot_history(history):

    acc = history.history['mean_absolute_error']
    val_acc = history.history['val_mean_absolute_error']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    fig, ax = plt.subplots(2, sharex = 'all')

    # Plot training and validation accuracy per epoch
    ax[0].plot(epochs, acc, label = 'Training')
    ax[0].plot(epochs, val_acc, label = 'Validation')
    ax[0].title.set_text('Training and validation MAE')

    # Plot training and validation loss per epoch
    ax[1].plot(epochs, loss, label = 'Training')
    ax[1].plot(epochs, val_loss, label = 'Validation')
    ax[1].title.set_text('Training and validation loss')

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'center right')
    plt.xticks(rotation=20)
    plt.xlabel('epochs')
    plt.show()


# ############# Model architecture #############

# TODO: Check input in flatten layer is adequate
# # Linear model
def f_linear_model(label_target, label_width):
    linear_model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(units = 40),
                    tf.keras.layers.Dense(units = 15),
                    tf.keras.layers.Dense(units = len(label_target) * label_width)])
    return linear_model

linear_model = f_linear_model(label_target, label_width)

# # MLP model
def f_mlp_model(label_target, label_width):
    mlp_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units = 65*10, activation = 'relu'),
                tf.keras.layers.Dense(units = 65*5, activation = 'relu'),
                tf.keras.layers.Dense(units = 65, activation = 'relu'),
                tf.keras.layers.Dense(units = len(label_target) * label_width)])
    return mlp_model
    
mlp_model = f_mlp_model(label_target, label_width)

# RNN - Vanilla
def f_rnn_model(input_shape, output_shape):
    rnn_model = tf.keras.Sequential([
                tf.keras.layers.SimpleRNN(units = 65 * 5, input_shape = input_shape),
                tf.keras.layers.Dense(output_shape)])
    return rnn_model

input_shape = train_data[0].shape
output_shape = train_target.shape[1]
rnn_model = f_rnn_model(input_shape, output_shape)


# # RNN - LSTM
# # lstm_model = 2


# # RNN - GRU
# # gru_model = 2


# ############# Deployment #############
# Callbacks
def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * np.exp(-0.1)

# Training
def compile_and_fit(model, train_data, train_target, val_data, val_target, test_data, test_target, max_epochs, batch_size, test_entry):

    # Add callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", 
                                                  patience = 4)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
    callbacks = [early_stop, lr_scheduler]

    # Model compile
    model.compile(loss      = tf.losses.MeanSquaredError(),
                  optimizer = tf.optimizers.Adam(),
                  metrics   = [tf.metrics.MeanAbsoluteError()])

    # Model fit
    history = model.fit(train_data,
                        train_target,
                        batch_size = batch_size,
                        epochs  = max_epochs,
                        verbose = 2,
                        shuffle = False, 
                        validation_data = (val_data, val_target),
                        callbacks = callbacks)

    model.evaluate(test_data, test_target)

    test_data = np.reshape(test_data[test_entry], (1, test_data[0].shape[0], test_data[0].shape[1]))
    test_prediction = model.predict(test_data)

    return history, test_prediction

max_epochs = 3
batch_size = 36
test_entry = np.random.randint(0, 100)

### Linear model evaluation
linear_history, linear_prediction = compile_and_fit(linear_model, train_data, train_target, val_data, val_target, test_data, test_target, 
                                                    max_epochs, batch_size, test_entry)

linear_results = linear_prediction.reshape(label_width, len(label_target))

plot_results(label_target, test_target_time[test_entry], test_target[test_entry], linear_results)

plot_history(linear_history)

### MLP model evaluation
mlp_history, mlp_prediction = compile_and_fit(mlp_model, train_data, train_target, val_data, val_target, test_data, test_target, 
                                                    max_epochs, batch_size, test_entry)

mlp_results = mlp_prediction.reshape(label_width, len(label_target))

plot_results(label_target, test_target_time[test_entry], test_target[test_entry], mlp_results)

plot_history(mlp_history)

### Simple RNN model evaluation=
rnn_history, rnn_prediction = compile_and_fit(rnn_model, train_data, train_target, val_data, val_target, test_data, test_target, 
                                                    max_epochs, batch_size, test_entry)

rnn_results = rnn_prediction.reshape(label_width, len(label_target))

plot_results(label_target, test_target_time[test_entry], test_target[test_entry], rnn_results)

plot_history(rnn_history)

