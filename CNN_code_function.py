# Based on code from Assignment 3 8P361
# TU/e BME Project Imaging 2021
# Convolutional neural network for PCAM
# Code by Suzanne Wetstein

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix



def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64, image_size=96):
    """
    Generates a convolutional neural network model with configurable parameters.

    Args:
        kernel_size (tuple): Tuple specifying the height and width of the 2D convolution window. Default is (3, 3).
        pool_size (tuple): Tuple specifying the factor by which to downscale in the pooling operation. Default is (4, 4).
        first_filters (int): Number of filters in the first convolutional layer. Default is 32.
        second_filters (int): Number of filters in the second convolutional layer. Default is 64.

    Returns:
        model: A Keras Sequential model.
    """
    # Build the model
    model = Sequential()

    # Add convolutional layers
    model.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    # Flatten the output to feed into dense layers
    model.add(Flatten())

    # Add dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(SGD(learning_rate=0.01, momentum=0.95), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(),'accuracy'])

    return model



def train_model(model, train_gen, val_gen, weights_filepath, model_name, epochs=1):
    """
    Trains the provided model using the given data generators and saves the best weights.

    Args:
        model: The Keras Sequential model to train.
        train_gen: Generator for training data.
        val_gen: Generator for validation data.
        weights_filepath (str): Filepath to save the best weights.
        model_name (str): Name of the model, used for TensorBoard logging.
        epochs (int): Number of epochs to train the model for. Default is 1.

    Returns:
        history: History object containing training metrics.
    """

    model_json = model.to_json() # serialize model to JSON
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)

    # Define callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir=os.path.join('logs', model_name))
    callbacks_list = [checkpoint, tensorboard]

    # Train the model
    train_steps = train_gen.n // train_gen.batch_size
    val_steps = val_gen.n // val_gen.batch_size

    history = model.fit(train_gen, steps_per_epoch=train_steps, 
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=epochs,
                        callbacks=callbacks_list)
    return history



def get_fcn_model(first_filters=32, second_filters=64, kernel_size=(3,3), pool_size=(4,4), image_size=96):
    """
    Generates a fully convolutional neural network model with configurable parameters.

    Args:
        first_filters (int): Number of filters in the first convolutional layer. Default is 32.
        second_filters (int): Number of filters in the second convolutional layer. Default is 64.
        kernel_size (tuple): Tuple specifying the height and width of the 2D convolution window. Default is (3, 3).
        pool_size (tuple): Tuple specifying the factor by which to downscale in the pooling operation. Default is (4, 4).

    Returns:
        model: A Keras Sequential model.
    """
    # Build the model
    model = Sequential()

    # Add convolutional layers
    model.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    # Output layer
    model.add(Conv2D(1, kernel_size, activation='sigmoid', padding='same'))
    model.add(GlobalAveragePooling2D())

    # Compile the model
    model.compile(SGD(learning_rate=0.01, momentum=0.95), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(),'accuracy'])

    return model



def calculate_roc_and_auc(model, val_gen):
    """
    Performs the calculations necessary for the ROC curve and its AUC value.

    Args:
        model: The trained Keras model.
        val_gen: Generator for validation data.

    Returns:
        tuple: False positive rate, true positive rate, and AUC value.
    """
    # Predict probabilities for the validation set
    predictions = model.predict(val_gen)

    # Calculate the false positive rate (FPR) and true positive rate (TPR)
    fpr, tpr, thresholds = roc_curve(val_gen.labels, predictions)
    
    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Print classification report
    predictions[predictions <= 0.5] = 0.
    predictions[predictions > 0.5] = 1.
    print('Classification report:\n', classification_report(val_gen.labels, predictions))
    print('Confusion matrix:\n', confusion_matrix(val_gen.labels, predictions))
    
    return fpr, tpr, roc_auc




def plot_roc_curve(fpr, tpr, roc_auc, type='dense'):
    """
    Plot the ROC curve using the false positive rate, true positive rate, and AUC value.

    Args:
        fpr (array): False positive rate.
        tpr (array): True positive rate.
        roc_auc (float): Area under the ROC curve.
        type (str): Type of model to use ('dense' or 'conv'). Default is 'dense'. 
    """
    # Plot the ROC curve
    if type == 'dense':
        name_type ='connected'
    elif type == 'conv':
        name_type='convolutional'
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='gray', lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve of model with fully {} layers".format(name_type))
    plt.legend(loc="lower right")
    plt.show()