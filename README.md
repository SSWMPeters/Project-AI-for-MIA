# VAE and CNN Model for Image Generation

This Python script contains functions for constructing, training, and evaluating Variational Autoencoder (VAE) and Convolutional Neural Network (CNN) models for the PatchCamelyon dataset.

## File Structure

- `VAE_code_function.py`: Contains functions related to VAE model construction, training, and image generation.
- `CNN_code_function.py`: Contains functions related to CNN model construction, training, and evaluation.
- `code_calls.ipynb`: Main script combining both VAE and CNN functionalities.

## Usage

1. **Setup**:
    - Ensure all necessary dependencies are installed. (`numpy`, `tensorflow`, etc.)
    - Set the base directory where the dataset is located (`base_dir`).
    - Define the model name (`model_name`).
    - Configure training parameters such as batch size (`batch_size`), image size (`img_size`), number of epochs (`nr_epochs`), split ratio for training data (`split`), and latent dimension for VAE (`latent_dim`).

2. **VAE Functionality**:
    - Run VAE training for specified classes by calling `vae.construct_vae(train_gen, val_gen, weights_filepath, model_name, nr_epochs, latent_dim)`.
    - Reconstruct and display images using `vae.reconstruct_img(model, val_gen)`.
    - Generate new images, display, and save them with `vae.generate_new_img(decoder, base_dir, latent_dim=latent_dim, split=split, class_type=i, num_samples=n)`.

3. **CNN Functionality**:
    - Use `calculate_and_plot_ROC_AUC()` to train a CNN model, calculate ROC curve, and plot AUC value.
    - Parameters: 
        - `base_dir`: Path to the dataset directory.
        - `split`: Ratio of training data.
        - `type`: Type of model ('dense' or 'conv').
        - `model_name`: Name of the model.

4. **Example Usage**:
    ```python
    import VAE_code_function as vae
    import CNN_code_function as cnn
    import numpy as np
    
    # Define parameters...
    
    # Perform VAE for specified classes
    classes = [0,1]
    for i in classes:
        train_gen, val_gen = vae.get_pcam_generators(base_dir, model_type='vae', split=split, batch_size_gen=batch_size, class_type=i)
        model, decoder = vae.construct_vae(train_gen, val_gen, weights_filepath, model_name, nr_epochs, latent_dim)
        vae.reconstruct_img(model, val_gen)
        vae.generate_new_img(decoder, base_dir, latent_dim=latent_dim, split=split, class_type=i, num_samples=n)
    
    # Calculate and plot ROC-AUC
    calculate_and_plot_ROC_AUC(base_dir, type='dense', split=split, model_name=model_name+'_dense')
    ```

## Requirements

- Python 3.x
- Dependencies: `numpy`, `tensorflow`, `sklearn`, `os`, `random`
