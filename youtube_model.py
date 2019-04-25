from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

print(tf.__version__)

column_names = ['Rank', 'Grade', 'Channel Name', 'Video Uploads', 'Subscribers',
                'Video Views']

raw_dataset = pd.read_csv('youtube_data.csv',
                          names=column_names,
                          na_values='?')

dataset = raw_dataset.copy()
dataset = dataset.drop([0]) #Drop first row?
channel_names = dataset.pop('Channel Name')
ranks = dataset.pop('Rank')
grade = dataset.pop('Grade')
dataset['Video Uploads'] = pd.to_numeric(dataset['Video Uploads'], errors='coerce')
dataset['Subscribers'] = pd.to_numeric(dataset['Subscribers'], errors='coerce')
dataset['Video Views'] = pd.to_numeric(dataset['Video Views'], errors='coerce')
grade_labels = ['B+ ', 'A- ', 'A ', 'A+ ', 'A++ ', 'A+++ ']

# One-hot encode the Grades column
#for i in range(1, 6):
#    dataset[grade_labels[i]] = (grade == grade_labels[i])*1.0

dataset['B+'] = (grade == grade_labels[0])*1.0
dataset['A-'] = (grade == grade_labels[1])*1.0
dataset['A'] = (grade == grade_labels[2])*1.0
dataset['A+'] = (grade == grade_labels[3])*1.0
dataset['A++'] = (grade == grade_labels[4])*1.0
#dataset['A+++'] = (grade == grade_labels[5])*1.0

dataset = dataset.dropna()


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('Video Views')
test_labels = test_dataset.pop('Video Views')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
                              layers.Dense(64, activation=tf.nn.relu),
                              layers.Dense(1)])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model()


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


EPOCHS = 10000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[early_stop, PrintDot()])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Video Views]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 2e+9])
    plt.legend()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Video Views^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 1e+19])
    plt.legend()
    plt.show()

plot_history(history)