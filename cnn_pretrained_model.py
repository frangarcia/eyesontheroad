# !pip install split_folders
import os
import splitfolders
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input

# Preproceso de datos.

if not os.path.exists('./datasets/dmd/labels_split'):
    splitfolders.ratio('./datasets/dmd/labels', './datasets/dmd/labels_split', seed=1, ratio=(.9, 0, .1))

INPUT_SHAPE = (224, 224, 3)
DATA_DIR = './datasets/dmd/labels_split/data'
BATCH_SIZE = 32
IMG_SIZE = (180, 180)
VAL_SPLIT = 0.2

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=VAL_SPLIT,
    subset='training',
    color_mode='rgb',
    seed=1,
    shuffle=0,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=VAL_SPLIT,
    subset='validation',
    color_mode='rgb',
    seed=1,
    shuffle=0,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print('Clases:' + class_names.join(','))


base_model = tf.keras.applications.VGG19(
    weights='imagenet',
    include_top=False
)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
preds = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)
print(model.summary())

print("Modelo base:", len(base_model.layers), "\nModelo:", len(model.layers))

for layer in model.layers[:19]:
    layer.trainable = False
for layer in model.layers[19:]:
    layer.trainable = True
model.summary()

model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy,
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

bestModel = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    epochs=10,
    callbacks=[early_stop]
)
