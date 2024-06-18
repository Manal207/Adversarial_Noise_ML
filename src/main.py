import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preparation import data_loader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
import tensorflow as tf

def preprocess_data(images, labels):
    # Normalize images
    images = images.astype('float32') / 255.0
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=10)
    
    return images, labels

def build_model():
    # Build a simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Load datasets
    train_images, train_labels = data_loader.load_dataset('data/cifar10/train')
    test_images, test_labels = data_loader.load_dataset('data/cifar10/test')
    
    # Preprocess datasets
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    
    # Build and train the model
    model = build_model()
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

if __name__ == "__main__":
    main()
