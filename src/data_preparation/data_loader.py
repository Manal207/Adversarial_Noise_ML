import os
import numpy as np
import pandas as pd
from PIL import Image

def load_dataset(data_dir):
    images = []
    labels = []
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    
    for index, row in labels_df.iterrows():
        image_path = os.path.join(data_dir, 'images', row['image_name'])
        image = Image.open(image_path)
        image = image.resize((32, 32))  # Ensure the image is the correct size
        images.append(np.array(image))
        labels.append(row['label'])
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# import os
# import numpy as np
# import pandas as pd
# from PIL import Image

# def load_dataset(data_dir):
#     images = []
#     labels = []
    
#     # Load the CSV file
#     labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    
#     # Load images and labels
#     for index, row in labels_df.iterrows():
#         image_path = os.path.join(data_dir, 'images', row['image_name'])
#         image = Image.open(image_path)
#         image = image.resize((32, 32))  # Ensure the image is the correct size
#         images.append(np.array(image))
#         labels.append(row['label'])
    
#     images = np.array(images)
#     labels = np.array(labels)
    
#     return images, labels

# # Load training and test datasets
# train_images, train_labels = load_dataset('data/cifar10/train')
# test_images, test_labels = load_dataset('data/cifar10/test')

# # Print shapes to verify
# print(f'Training images shape: {train_images.shape}')
# print(f'Training labels shape: {train_labels.shape}')
# print(f'Test images shape: {test_images.shape}')
# print(f'Test labels shape: {test_labels.shape}')
