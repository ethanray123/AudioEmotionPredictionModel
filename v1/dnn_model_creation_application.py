# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv
import pandas as pd
# import seaborn as sns
# import skfuzzy as sf
# from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

# VALENCE
train_path_valence = 'train_data_valence.csv'
test_path_valence = 'test_data_valence.csv'
train_data = np.genfromtxt(train_path_valence, delimiter=',', skip_header=1)
x_train, y_train = train_data[:, 0:-1], train_data[:, -1]
y_train = y_train.astype(int)
test_data = np.genfromtxt(test_path_valence, delimiter=',', skip_header=1)
x_test, y_test = test_data[:, 0:-1], test_data[:, -1]
y_test = y_test.astype(int)
print("x_train: {}".format(x_train.shape))
print("y_train: {}".format(y_train.shape))
print("x_test: {}".format(x_test.shape))
print("y_test: {}".format(y_test.shape))
x_train = tf.keras.utils.normalize(
    x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(
    x_test, axis=1)  # scales data between 0 and 1
# print("{} - {}".format(data.shape, type(data)))

v_model = tf.keras.models.Sequential()  # a basic feed-forward model
v_model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
# a simple fully-connected layer, 128 units, relu activation
v_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# a simple fully-connected layer, 128 units, relu activation
v_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# our output layer. 10 units for 10 classes. Softmax for probability distribution
v_model.add(tf.keras.layers.Dense(9, activation=tf.nn.softmax))

v_model.compile(optimizer='adam',  # Good default optimizer to start with
                # how will we calculate our "error." Neural network aims to minimize loss.
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])  # what to track

v_model.fit(x_train, y_train, epochs=3)  # train the model

# evaluate the out of sample data with model
val_loss, val_acc = v_model.evaluate(x_test, y_test)
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy

v_model.save('audio_nn_valence_classifier.model')
loaded_v_model = tf.keras.models.load_model(
    'audio_nn_valence_classifier.model')
predictions = loaded_v_model.predict(x_test)

print("Valence Prediction: {}".format(np.argmax(predictions[0])))
print("Expected Valence Output: {}".format(y_test[0]))
# for i in range(len(y_test)):
#     print("Valence Prediction: {}".format(np.argmax(predictions[i])))
#     print("Expected Valence Output: {}".format(y_test[i]))


# AROUSAL
train_path_arousal = 'train_data_arousal.csv'
test_path_arousal = 'test_data_arousal.csv'
train_data = np.genfromtxt(train_path_arousal, delimiter=',', skip_header=1)
x_train, y_train = train_data[:, 0:-1], train_data[:, -1]
y_train = y_train.astype(int)
test_data = np.genfromtxt(test_path_arousal, delimiter=',', skip_header=1)
x_test, y_test = test_data[:, 0:-1], test_data[:, -1]
y_test = y_test.astype(int)
print("x_train: {}".format(x_train.shape))
print("y_train: {}".format(y_train.shape))
print("x_test: {}".format(x_test.shape))
print("y_test: {}".format(y_test.shape))
x_train = tf.keras.utils.normalize(
    x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(
    x_test, axis=1)  # scales data between 0 and 1
# print("{} - {}".format(data.shape, type(data)))

a_model = tf.keras.models.Sequential()  # a basic feed-forward model
a_model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
# a simple fully-connected layer, 128 units, relu activation
a_model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
# a simple fully-connected layer, 128 units, relu activation
a_model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
# our output layer. 10 units for 10 classes. Softmax for probability distribution
a_model.add(tf.keras.layers.Dense(9, activation=tf.nn.softmax))

a_model.compile(optimizer='adam',  # Good default optimizer to start with
                # how will we calculate our "error." Neural network aims to minimize loss.
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])  # what to track

a_model.fit(x_train, y_train, epochs=3)  # train the model

# evaluate the out of sample data with model
val_loss, val_acc = a_model.evaluate(x_test, y_test)
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy

a_model.save('audio_nn_arousal_classifier.model')
loaded_a_model = tf.keras.models.load_model(
    'audio_nn_arousal_classifier.model')
predictions = loaded_a_model.predict(x_test)

print("Arousal Prediction: {}".format(np.argmax(predictions[0])))
print("Expected Arousal Output: {}".format(y_test[0]))
# for i in range(len(y_test)):
#     print("Arousal Prediction: {}".format(np.argmax(predictions[i])))
#     print("Expected Arousal Output: {}".format(y_test[i]))
