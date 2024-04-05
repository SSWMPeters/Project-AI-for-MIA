# VAE and CNN Model for Image Generation

These files contain functions for constructing, training, and evaluating Variational Autoencoder (VAE) and Convolutional Neural Network (CNN) models for the PatchCamelyon dataset.

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

## Requirements

- Python 3.x
- Dependencies: `numpy`, `tensorflow`, `sklearn`, `os`, `random`
