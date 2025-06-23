import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import random



#displaay dog image
Dogs = '/Users/nithunsundarrajan/Downloads/PetImages/Dog'
Cats =  '/Users/nithunsundarrajan/Downloads/PetImages/Cat'

X = []
Y = []


image_Size = (64,64)

for filename in os.listdir(Dogs):
    if filename.endswith('.jpg'):
        path = os.path.join(Dogs, filename)
        img = Image.open(path).convert('RGB')  # Ensure the image is in RGB format
        img = img.resize(image_Size)
        X.append(np.array(img))
        Y.append(1)

for filename in os.listdir(Cats):
    if filename.endswith('.jpg'):
        path = os.path.join(Cats, filename)
        img = Image.open(path).convert('RGB')  # Ensure the image is in RGB format
        img = img.resize(image_Size)
        X.append(np.array(img))
        Y.append(0)

X = np.array(X)/255.0
Y = np.array(Y)

combined = list(zip(X, Y))
random.shuffle(combined)

X, Y = zip(*combined)  # we are basically converting it to lists , shuffling it and then converting it back to numpy arrays

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64,3)),  # Flatten the 2D images to 1D
    keras.layers.Dense(80, activation='relu'),  # Fully connected layer with 128 neurons
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(80, activation='relu'),  # Fully connected layer with 128 neurons
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'), keras.layers.Dense(80, activation='relu'),  # Fully connected layer with 128 neurons
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'), keras.layers.Dense(80, activation='relu'),  # Fully connected layer with 128 neurons
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')   # Output layer with 10 neurons for 10 classes

])


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train , epochs=20)






