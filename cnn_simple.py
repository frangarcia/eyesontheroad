import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import splitfolders
import os
import pickle

UNSPLITTED_FOLDER_IMAGES = './datasets/dmd/binary_labels'
SPLITTED_FOLDER_IMAGES = './datasets/dmd/binary_labels_split'

if not os.path.exists(SPLITTED_FOLDER_IMAGES):
    splitfolders.ratio(UNSPLITTED_FOLDER_IMAGES, SPLITTED_FOLDER_IMAGES, seed=1, ratio=(.7, .2, .1))

if not os.path.exists(SPLITTED_FOLDER_IMAGES):
    splitfolders.ratio(UNSPLITTED_FOLDER_IMAGES, SPLITTED_FOLDER_IMAGES, seed=1, ratio=(.7, .2, .1))

INPUT_SHAPE = (180, 180, 3)
TRAIN_DIR = SPLITTED_FOLDER_IMAGES + '/train'
VAL_DIR = SPLITTED_FOLDER_IMAGES + '/val'
TEST_DIR = SPLITTED_FOLDER_IMAGES + '/test'
BATCH_SIZE = 32
IMG_SIZE = (180, 180)
VAL_SPLIT = 0.2

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    validation_split=VAL_SPLIT,
    subset='training',
    color_mode='rgb',
    seed=1,
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    validation_split=VAL_SPLIT,
    subset='validation',
    color_mode='rgb',
    seed=1,
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

from keras.applications.vgg19 import preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

MODEL_SAVE_FOLDER = './models/cnn/cnn_simple/'
if os.path.exists(MODEL_SAVE_FOLDER):
    model = keras.models.load_model(MODEL_SAVE_FOLDER + 'best')
else:
    model = keras.models.Sequential([
        Input(shape=INPUT_SHAPE),

        Conv2D(filters=32, kernel_size=5, strides=1, activation='relu'),
        Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', use_bias=False),
        MaxPooling2D(strides=2),
        Dropout(0.3),

        Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
        Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', use_bias=False),
        MaxPooling2D(strides=2),
        Dropout(0.3),

        Flatten(),
        Dense(units=256, activation='relu', use_bias=False),

        Dense(units=128, use_bias=False, activation='relu'),

        Dense(units=84, use_bias=False, activation='relu'),
        Dropout(0.3),

        Dense(units=num_classes, activation='softmax')
    ])

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    best_model_checkpoint = keras.callbacks.ModelCheckpoint(MODEL_SAVE_FOLDER + 'best', monitor='val_loss',
                                                            save_best_only=True, mode='min')
    history = model.fit(x=train_ds, validation_data=val_ds, epochs=10,
                        callbacks=[early_stop, best_model_checkpoint])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_SAVE_FOLDER + 'model.tflite', 'wb') as f:
        f.write(tflite_model)

    with open(MODEL_SAVE_FOLDER + 'model_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

with open(MODEL_SAVE_FOLDER + 'model_history.pkl', 'rb') as file:
    history = pickle.load(file)

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Entrenamiento acc')
plt.plot(epochs, val_acc, 'b', label='Validación acc')
plt.title('Accuracy - exactitud de entrenamiento y validación')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Entrenamiento loss')
plt.plot(epochs, val_loss, 'b', label='Validación loss')
plt.title('Loss - función objetivo en entrenamiento y prueba')
plt.legend()

plt.show()
