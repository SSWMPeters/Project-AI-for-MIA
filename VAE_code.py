from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.losses import mse
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Reshape, Conv2DTranspose
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import random

def get_pcam_generators(base_dir, class_type, batch_size_gen=32, split=0.2, img_size=96):
    """
    Uses the ImageDataGenerator function from the Keras API to return images in batches,
    train_gen for the training data and val_gen for the validation data.

    Args:
        base_dir (str): Base directory containing the dataset.
        train_batch_size (int): Batch size for training data. Default is 32.
        val_batch_size (int): Batch size for validation data. Default is 32.

    Returns:
        tuple: A tuple containing train_gen and val_gen, both are generators.
    """
    # Dataset parameters
    TRAIN_PATH = os.path.join(base_dir, 'train+val', 'train')
    VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')
    RESCALING_FACTOR = 1./255

    split = 1-split
    
    # Instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR,
                                 validation_split=split)

    # Generate data batches for training and validation sets
    train_gen = datagen.flow_from_directory(TRAIN_PATH,
                                            target_size=(img_size, img_size),
                                            batch_size=batch_size_gen,
                                            subset='training',
                                            classes=[str(class_type)],
                                            class_mode='input')

    val_gen = datagen.flow_from_directory(VALID_PATH,
                                          target_size=(img_size, img_size),
                                          batch_size=batch_size_gen,
                                          class_mode='input',
                                          classes=[str(class_type)],
                                          shuffle=False)
    return train_gen, val_gen

def construct_vae(train_gen, val_gen, weights_filepath, model_name, nr_epochs=1, latent_dim=50, batch_size=32, img_size=96):
    
    # Define input shape and latent dimension
    input_shape = (img_size, img_size, 3)
    
    # Encoder network
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    shape_before_flattening = K.int_shape(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    

    # Sampling function
    @tf.function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(z_log_var / 2) * epsilon
    

    # Reparameterization trick
    z = Lambda(sampling)([z_mean, z_log_var])
    

    # Decoder network
    decoder_input = Input(K.int_shape(z)[1:])
    x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = Reshape(shape_before_flattening[1:])(x)
    x = Conv2DTranspose(128, (2, 2), activation='relu', padding='same', )(x)
    x = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2DTranspose(32, (2, 2), activation='relu', padding='same', )(x)
    x = Conv2DTranspose(16, (2, 2), activation='relu', padding='same', )(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Define the VAE model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, x, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name=model_name)

    # Define the VAE loss function
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
    B = 1000   
    vae_loss = K.mean(B * reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.add_metric(kl_loss, name="kl_loss")
    vae.add_metric(reconstruction_loss, name="reconstruction_loss")
    vae.compile(optimizer='adam')

    # Serialize model to JSON
    model_json = vae.to_json() # serialize model to JSON
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)
    
    # Define EarlyStopping callback to stop training when the model stops improving
    early_stopping = EarlyStopping(monitor="val_loss", 
                                   patience=1,
                                   verbose=1,
                                   mode='min',
                                   restore_best_weights=True)
    
    # Define other callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir=os.path.join('logs', model_name))
    callbacks_list = [checkpoint, tensorboard, early_stopping]

    # Calculate number of steps per epoch for training and validation sets
    train_steps = train_gen.n // train_gen.batch_size
    val_steps = val_gen.n // val_gen.batch_size
    
    # Fit the VAE model
    vae.fit(train_gen, 
            steps_per_epoch=train_steps, 
            epochs=nr_epochs, 
            batch_size=batch_size, 
            validation_data=val_gen, 
            validation_steps=val_steps,
            callbacks=callbacks_list,
            verbose=1)

    # Returns the trained VAE model
    return vae, decoder

def reconstruct_img(vae, val_gen, img_size=96):
    
    # Reconstruct images using the trained VAE model
    decoded_imgs = vae.predict(val_gen)
    
    n = 10
    # Display the original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(n):
        
        # Display the original image
        ax = plt.subplot(2, n, i + 1)
        img, label = val_gen.next()
        plt.imshow(img[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display the reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(img_size, img_size,3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def display_generated(model, random_vector):
    n = 10 # number of images to display
    new_image = model.predict(random_vector)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display the reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        image = new_image.squeeze()[i]/new_image.squeeze()[i].max()
        plt.imshow(image)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def generate_new_img(vae, base_dir, class_type, split, num_samples=10, delete_files=True, img_size=96):
    
    # Generate new images using random latent vectors
    random_latent_vectors  = np.random.random((num_samples, img_size, img_size, 3))
    decoded_imgs = vae.predict(random_latent_vectors)
    
    # Path to the directory where you want to save the images
    save_dir = base_dir + "/train_new_images_{}/{}/".format(str(split*100), str(class_type))

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Optionally delete existing files in the directory
    for file in os.listdir(save_dir):
        if file.endswith('.jpg') and delete_files == True:
            os.remove(save_dir + file)

    # Iterate through generated images and save them
    for i in range(len(decoded_imgs)):
        img = decoded_imgs[i].reshape(img_size, img_size,3)
        
        # Generate a random name for the image
        random_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=10))
        
        # Save the image with the random name and jpg extension
        plt.imsave(os.path.join(save_dir, random_name + ".jpg"), img)

    print("Images saved to:", save_dir)