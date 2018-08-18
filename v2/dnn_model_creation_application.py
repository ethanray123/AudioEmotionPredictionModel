# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import csv

# LOAD AND NORMALIZE DATA
x_train = np.load("train_x.npy")
valence_y_train = np.load("valence_train_y.npy")
arousal_y_train = np.load("arousal_train_y.npy")
x_test = np.load("test_x.npy")
valence_y_test = np.load("valence_test_y.npy")
arousal_y_test = np.load("arousal_test_y.npy")

print("x_train: {}".format(x_train.shape))
print("valence_y_train: {}".format(valence_y_train.shape))
print("arousal_y_train: {}".format(arousal_y_train.shape))
print("x_test: {}".format(x_test.shape))
print("valence_y_test: {}".format(valence_y_test.shape))
print("arousal_y_test: {}".format(arousal_y_test.shape))
# print("{} - {}".format(x_train[0][1], type(x_train)))
x_train = tf.keras.utils.normalize(
    x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(
    x_test, axis=1)  # scales data between 0 and 1
# print("{} - {}".format(x_train[0][1], type(x_train)))

# CONFIGURE MODELS AND EVALUATE LOSS AND ACCURACY
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

v_model.fit(x_train, valence_y_train, epochs=3)  # train the model

# evaluate the out of sample data with model
v_loss, v_acc = v_model.evaluate(x_test, valence_y_test)

a_model = tf.keras.models.Sequential()  # a basic feed-forward model
a_model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
# a simple fully-connected layer, 128 units, relu activation
a_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# a simple fully-connected layer, 128 units, relu activation
a_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# our output layer. 10 units for 10 classes. Softmax for probability distribution
a_model.add(tf.keras.layers.Dense(100, activation=tf.nn.softmax))

a_model.compile(optimizer='adam',  # Good default optimizer to start with
                # how will we calculate our "error." Neural network aims to minimize loss.
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])  # what to track

a_model.fit(x_train, arousal_y_train, epochs=3)  # train the model

# evaluate the out of sample data with model
a_loss, a_acc = a_model.evaluate(x_test, arousal_y_test)

# LOSS AND ACCURACIES
print("\n\nValence Model's loss: {}".format(v_loss))  # model's loss (error)
print("Valence Model's accuracy: {}\n\n".format(v_acc))  # model's accuracy
print("\n\nArousal Model's loss: {}".format(a_loss))  # model's loss (error)
print("Arousal Model's accuracy: {}\n\n".format(a_acc))  # model's accuracy

# LOAD MDOELS AND PREDICT AROUSAL AND VALENCE FROM TEST VALUES
v_model.save('audio_nn_valence_classifier.model')
loaded_v_model = tf.keras.models.load_model(
    'audio_nn_valence_classifier.model')
v_predictions = loaded_v_model.predict(x_test)

a_model.save('audio_nn_arousal_classifier.model')
loaded_a_model = tf.keras.models.load_model(
    'audio_nn_arousal_classifier.model')
a_predictions = loaded_a_model.predict(x_test)

print("Valence Prediction: {}".format(np.argmax(v_predictions[0])))
print("Expected Valence Output: {}".format(valence_y_test[0]))

print("Arousal Prediction: {}".format(np.argmax(a_predictions[0])))
print("Expected Arousal Output: {}".format(arousal_y_test[0]))

# for i in range(len(valence_y_test)):
#     print("Valence Prediction: {}".format(np.argmax(v_predictions[i])))
#     print("Expected Valence Output: {}".format(valence_y_test[i]))

# for i in range(len(arousal_y_test)):
#     print("Arousal Prediction: {}".format(np.argmax(a_predictions[i])))
#     print("Expected Arousal Output: {}".format(arousal_y_test[i]))
