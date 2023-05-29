import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model  # Update import statement
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout  # Update import statement
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

UNSPLITTED_FOLDER_IMAGES = './datasets/dmd/binary_labels'
SPLITTED_FOLDER_IMAGES = './datasets/dmd/binary_labels_split'
INPUT_SHAPE = (180, 180, 3)
TRAIN_DIR = SPLITTED_FOLDER_IMAGES + '/train'
VAL_DIR = SPLITTED_FOLDER_IMAGES + '/val'
TEST_DIR = SPLITTED_FOLDER_IMAGES + '/test'
BATCH_SIZE = 32
IMG_SIZE = (180, 180)
VAL_SPLIT = 0.2

if not os.path.exists(SPLITTED_FOLDER_IMAGES):
    splitfolders.ratio(UNSPLITTED_FOLDER_IMAGES, SPLITTED_FOLDER_IMAGES, seed=1, ratio=(.7, .2, .1))

X = []
Y = []

for i in tqdm(glob(UNSPLITTED_FOLDER_IMAGES + '/awake/*')):
    temp = np.array(Image.open(i).resize(IMG_SIZE))
    X.append(temp)
    Y.append(1)

for i in tqdm(glob(UNSPLITTED_FOLDER_IMAGES + '/drowsy/*')):
    temp = np.array(Image.open(i).resize(IMG_SIZE))
    X.append(temp)
    Y.append(0)

X = np.array(X)
X = X / 255.0
Y = np.array(Y)
X = np.expand_dims(X, -1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

MODEL_SAVE_FOLDER = './models/cnn/cnn_simple/'
if os.path.exists(MODEL_SAVE_FOLDER):
    model = keras.models.load_model(MODEL_SAVE_FOLDER + 'best')
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

    model.summary()

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
    modelFit = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, model_checkpoint]
    )

    model.evaluate(x_test, y_test)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_SAVE_FOLDER + 'model.tflite', 'wb') as f:
        f.write(tflite_model)

    with open(MODEL_SAVE_FOLDER + 'model_history.pkl', 'wb') as file:
        pickle.dump(modelFit.history, file)

with open(MODEL_SAVE_FOLDER + 'model_history.pkl', 'rb') as file:
    history = pickle.load(file)

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, label='Entrenamiento acc', color='green')
plt.plot(epochs, val_acc, label='Validación acc')
plt.title('Accuracy - exactitud de entrenamiento y validación')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Entrenamiento loss')
plt.plot(epochs, val_loss, 'b', label='Validación loss')
plt.title('Loss - función objetivo en entrenamiento y prueba')
plt.legend()

plt.show()

from keras.models import load_model

best_model = load_model(MODEL_SAVE_FOLDER + 'best.h5')
best_model.evaluate(x_test, y_test)

for i in x_test[0:5]:
    result = best_model.predict(np.expand_dims(i, 0))
    plt.imshow(i)
    plt.show()

    if result > 0.5:
        print('Awake')
    else:
        print("Drowsy")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

plt.figure(figsize=(15, 5))

preds = best_model.predict(x_test)
preds = (preds >= 0.5).astype(np.int32)
cm = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(cm, index=['closed', 'Open'], columns=['Closed', 'Open'])
plt.subplot(121)
plt.title("Confusion matrix\n")
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.ylabel("Predicted")
plt.xlabel("Actual")
