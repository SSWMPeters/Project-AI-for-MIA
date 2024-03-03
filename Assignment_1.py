from PIL import Image
import os
import matplotlib.pyplot as plt

def display_images(image_paths, titles):
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 6))
    for i in range(num_images):
        img = plt.imread(image_paths[i])
        axes[i].imshow(img)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.show()


def select_images(directory, num_images=3):
    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(('.tif','.jpg'))]
    class_images = []
    # Display the first three images
    for i in range(min(num_images, len(image_files))):
        # Open the image file
        image_path = os.path.join(directory, image_files[i])
        class_images.append(image_path)
        print(image_path)
    return class_images

# Replace 'directory' with the path to your directory containing the image files
directory_0 = 'C:/Users/20213002/.vscode/Y3Q3 Project AI MIA/train/0'
directory_1 = 'C:/Users/20213002/.vscode/Y3Q3 Project AI MIA/train/1'

paths_0 = select_images(directory_0)
paths_1 = select_images(directory_1)

display_images(paths_0, ['Class 0: Patch 1', 'Class 0: Patch 2', 'Class 0: Patch 3'])
display_images(paths_1, ['Class 1: Patch 1', 'Class 1: Patch 2', 'Class 1: Patch 3'])