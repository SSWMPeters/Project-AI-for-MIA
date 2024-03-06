'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary',
                                             shuffle=False)

     return train_gen, val_gen


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model



def get_model_2(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):
    # build the model
    model = Sequential()

    model.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    # Convolutional layer with 64 filters and ReLU activation
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))    

    # Convolutional layer with 1 filter and sigmoid activation
    model.add(Conv2D(1, kernel_size, activation='sigmoid', padding='same'))
    model.add(Flatten())

    # compile the model
    model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

    return model


# get the model
model = get_model_2()


# get the data generators
train_gen, val_gen = get_pcam_generators('C:/Users/20213002/.vscode/Y3Q3 Project AI MIA')

for layer in model.layers:
    print(layer.output_shape)

# save the model and weights
model_name = 'my_second_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)

# ROC analysis
model.load_weights(weights_filepath)

score = model.evaluate(val_gen)
print("Loss: ",score[0])
print("Accuracy: ",score[1])

# TODO Perform ROC analysis on the validation set
def calculate_roc_and_auc(model, val_gen):
    # Predict probabilities for the validation set
    predictions = model.predict(val_gen)
    
    # Calculate the false positive rate (FPR) and true positive rate (TPR)
    fpr, tpr, thresholds = roc_curve(val_gen.classes, predictions)
    
    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color = 'gray', lw = 2, label = "ROC curve (area = %0.2f)"  % roc_auc)
    plt.plot([0, 1], [0,1], color = 'gray', linestyle = '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc = "lower right")
    plt.show()

fpr, tpr, roc_auc = calculate_roc_and_auc(model, val_gen)
    
    # plot ROC curve
plot_roc_curve(fpr, tpr, roc_auc)
    
print("AUC", roc_auc)
