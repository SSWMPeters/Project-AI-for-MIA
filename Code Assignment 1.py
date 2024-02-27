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

# Directory containing images for class 1
class_1_dir = r'C:\Users\20203894\Documents\8p361\train+val\train\0'
class_1_images = [f'{class_1_dir}\\0000d563d5cfafc4e68acb7c9829258a298d9b6a.jpg',
    f'{class_1_dir}\\000aa638312a3dad22ef04b8a7df3fc98fc2e7c3.jpg',
    f'{class_1_dir}\\000d4bcc9d239e8304890ffd764794e93504e475.jpg']

# Directory containing images for class 2
class_2_dir = r'C:\Users\20203894\Documents\8p361\train+val\train\1'
class_2_images = [f'{class_2_dir}\\0000da768d06b879e5754c43e2298ce48726f722.jpg',
    f'{class_2_dir}\\000a2a35668f04edebc0b06d5d133ad90c93a044.jpg',
    f'{class_2_dir}\\000aa5d8f68dc1f45ebba53b8f159aae80e06072.jpg']

# Display images with titles
display_images(class_1_images, ['Class 1: Patch 1', 'Class 1: Patch 2', 'Class 1: Patch 3'])
display_images(class_2_images, ['Class 2: Patch 1', 'Class 2: Patch 2', 'Class 2: Patch 3'])
