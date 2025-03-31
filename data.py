import pandas as pd
import numpy as np
from PIL import Image
import gzip
import os
import re
from sklearn.model_selection import train_test_split


def load_tabular_data(filepath):
    """Loads tabular data (either .csv or .txt)"""
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".txt"):
        return pd.read_table(filepath, delimiter=' ')

def load_images(folder_path):
    """
    Loads data from mnist dataset
    Returns:
        - X_train: list of arrays representing digits
        - y_train: array of target labels

    """
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            npz_path = os.path.join(folder_path, filename)
            with np.load(npz_path) as data:
                print("Available keys in npz file: ", data.files)

                # find and load arrays based on suspected key patterns
                X_train = data['X_train']
                y_train = data['y_train']
            # Combine training and testing sets to be split later on
          
    
    return X_train, y_train


X_train, y_train = load_images("./data")



def preprocess_images(images, convert_gray=False, normalize=True, rotate=0):
    """Preprocess a list of images (numpy arrays), including normalization and optional conversion to grayscale."""
    processed_images = []
    for img in images:
        if convert_gray:
            img = Image.fromarray(img).convert('L')  # Convert RGB to Grayscale
        else:
            img = Image.fromarray(img)
        
        if rotate != 0:
            img = img.rotate(rotate)
        
        img = np.array(img)
        if normalize:
            img = img / 255.0  
        processed_images.append(img)
    return np.array(processed_images)


def load_text_data(filepath):
    """Loads text data from any text file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    """Preprocess text data by cleaning and encoding."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # One-hot encoding or any other encoding should typically be done after tokenization, so it might require a different setup.
    return text


def split_data(data, labels, train_size=0.7, validation_size=0.15, test_size=0.15, random_state=6600):
    """Splits data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, train_size=train_size, random_state=random_state)
    test_val_size = test_size / (validation_size + test_size)  # Correct split ratio
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data(data, filepath):
    """Saves processed data to disk."""
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, np.ndarray):
        np.save(filepath, data)
    else:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(str(data))