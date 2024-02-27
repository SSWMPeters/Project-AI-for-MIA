"""
TU/e BME Project Imaging 2021
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# select if numbers are grouped by shape or not
group_classes = True

# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255


def transform_categories(array):
    categories = {
        1: 0,  # vertical digits
        7: 0,  # vertical digits
        0: 1,  # loopy digits
        6: 1,  # loopy digits
        8: 1,  # loopy digits
        9: 1,  # loopy digits
        2: 2,  # curly digits
        5: 2,  # curly digits
        3: 3,  # other
        4: 3   # other
    }
    # Create a NumPy array with category integers corresponding to the values in 'array'
    categorized_array = np.vectorize(categories.get)(array)
    return categorized_array


if group_classes == True:
    y_train = transform_categories(y_train)
    y_val = transform_categories(y_val)
    y_test = transform_categories(y_test)

# define number of categories
num_classes = max(y_train)+1

# convert 1D class arrays to 10D class matrices
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

def plt_classes(y, num_class=10):
    plt.figure()
    plt.hist(y, bins=range(0,num_class+1), align='left', rwidth=0.9)
    plt.xlabel('Class')
    plt.ylabel('Class count')
    plt.xticks(range(0,num_class))
    plt.title('Class distribution')

def exercise_1(nr_neurons=64, activation_type='relu', num_classes=10):
    """This function determines the characteristics of the neural network.
    The number of neurons and activation type can be adjusted for three hidden layers."""
    model = Sequential()
    # flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
    model.add(Flatten(input_shape=(28,28,1))) 
    # fully connected layers
    model.add(Dense(nr_neurons, activation=activation_type))
    model.add(Dense(nr_neurons, activation=activation_type))
    model.add(Dense(nr_neurons, activation=activation_type))
    # output layer with 10 nodes (one for each class) and softmax nonlinearity
    model.add(Dense(num_classes, activation='softmax')) 
    return model

def exercise_2(excercise_nr=1, num_classes=10):
    """This function determines the characteristics of the neural network.
    Depending on the parameter of excercise_nr, the neural network yields a
    different responds. Function returns the model which has to be processed further.
    args:
        excercise_nr: if 1 --> No hidden layers are applied
        excersice_nr: if 2 --> Three hidden layers with 64 neurons and ReLu nonlinearity
        excersice_nr: if 3 --> Three hidden layers with 64 neurons and linear activation function"""
    
    model = Sequential()
    # flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
    model.add(Flatten(input_shape=(28,28,1))) 
    if excercise_nr == 2:
        # fully connected layer with 64 neurons and ReLU nonlinearity
        model.add(Dense(64, activation='relu'))
        # fully connected layer with 64 neurons and ReLU nonlinearity
        model.add(Dense(64, activation='relu'))
        # fully connected layer with 64 neurons and ReLU nonlinearity
        model.add(Dense(64, activation='relu'))
    elif excercise_nr == 3:
        # fully connected layer with 64 neurons and linear activation funtion
        model.add(Dense(64, activation='linear'))
        # fully connected layer with 64 neurons and linear activation funtion
        model.add(Dense(64, activation='linear'))
        # fully connected layer with 64 neurons and linear activation funtion
        model.add(Dense(64, activation='linear'))
    # output layer with 10 nodes (one for each class) and softmax nonlinearity
    model.add(Dense(num_classes, activation='softmax')) 
    return model

model = exercise_1(nr_neurons=128, activation_type='relu', num_classes=num_classes)
# model = exercise_2(excercise_nr=3, num_classes=num_classes)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

# use this variable to name your model
model_name="my_first_model"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/" + model_name)

# train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])


score = model.evaluate(X_test, y_test, verbose=0)
print("Loss: ",score[0])
print("Accuracy: ",score[1])

# predict classes for model
y_pred = model.predict(X_test) 
y_pred = np.argmax(y_pred, axis=1)

# convert one-hot to index
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_pred, y_test))

# display confustion matrix
cm = confusion_matrix(y_pred, y_test, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=list(range(num_classes)))
disp.plot()

# display distribution of classes
plt_classes(y_pred, num_classes)
plt.show()

