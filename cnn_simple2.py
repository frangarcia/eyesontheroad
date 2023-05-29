import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import os
import pickle
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

FOLDER_IMAGES = './datasets/dmd/binary_labels'
INPUT_SHAPE = (180, 180, 3)
BATCH_SIZE = 32
IMG_SIZE = (180, 180)
VAL_SPLIT = 0.2
MODEL_SAVE_FOLDER = './models/cnn/cnn_simple/'

X = []
Y = []
for i in tqdm(glob(FOLDER_IMAGES + '/awake/*')):
    temp = np.array(Image.open(i).resize(IMG_SIZE))
    X.append(temp)
    Y.append(1)

for i in tqdm(glob(FOLDER_IMAGES + '/drowsy/*')):
    temp = np.array(Image.open(i).resize(IMG_SIZE))
    X.append(temp)
    Y.append(0)

X = np.array(X)
X = X / 255.0
Y = np.array(Y)
X = np.expand_dims(X, -1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

if os.path.exists(MODEL_SAVE_FOLDER):
    model = keras.models.load_model(MODEL_SAVE_FOLDER + 'best.h5')
else:
    model = keras.models.Sequential([
        Input(shape=INPUT_SHAPE),

        Conv2D(filters=32, kernel_size=5, strides=1, activation='relu'),
        Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', use_bias=False),
        BatchNormalization(),
        MaxPooling2D(strides=2),
        Dropout(0.3),

        Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
        Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', use_bias=False),
        BatchNormalization(),
        MaxPooling2D(strides=2),
        Dropout(0.3),

        Flatten(),
        Dense(units=256, activation='relu', use_bias=False),
        BatchNormalization(),

        Dense(units=128, use_bias=False, activation='relu'),

        Dense(units=84, use_bias=False, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(units=1, activation='sigmoid')
    ])

    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_FOLDER + 'best.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, model_checkpoint]
    )

    with open(MODEL_SAVE_FOLDER + 'model_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

with open(MODEL_SAVE_FOLDER + 'model_history.pkl', 'rb') as file:
    history = pickle.load(file)
    print(history)

best_model = load_model(MODEL_SAVE_FOLDER + 'best.h5')
best_model.evaluate(x_test, y_test)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(MODEL_SAVE_FOLDER + 'best.tflite', 'wb') as f:
    f.write(tflite_model)
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, label='Entrenamiento acc', color='green')
plt.plot(epochs, val_acc, label='Validación acc', color='blue')
plt.title('Accuracy - exactitud de entrenamiento y validación')
plt.legend()
plt.figure()

plt.plot(epochs, loss, color='green', label='Entrenamiento loss')
plt.plot(epochs, val_loss, color='blue', label='Validación loss')
plt.title('Loss - función objetivo en entrenamiento y prueba')
plt.legend()

plt.show()