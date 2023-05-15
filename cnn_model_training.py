#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import keras_tuner as kt

# In[2]:


INPUT_SHAPE = (224, 224, 3)
DATA_DIR = './datasets/dmd/labels'
BATCH_SIZE = 32
IMG_SIZE = (180, 180)
VALIDATION_SPLIT = 0.2


# In[3]:


(train_ds, val_ds) = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=VALIDATION_SPLIT,
    subset='both',
    color_mode='grayscale',
    seed=2,
    shuffle=0,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)


# In[4]:


class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)


# In[5]:


layers_data_augmentation = keras.Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.3, fill_mode='constant'),
        preprocessing.RandomZoom(0.2),
        keras.layers.RandomBrightness(factor=0.2)]
)


# In[6]:


def model_builder(hp):
    model = keras.Sequential();
    # model.add(layers_data_augmentation)
    model.add(keras.layers.Flatten(input_shape=IMG_SIZE))

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1_nodes = hp.Int('layer_1_nodes', min_value=32, max_value=128, step=16)
    hp_layer_2_nodes = hp.Int('layer_2_nodes', min_value=32, max_value=128, step=16)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.add(keras.layers.Dense(units=hp_layer_1_nodes, activation=hp_activation))
    model.add(keras.layers.Dense(units=hp_layer_2_nodes, activation=hp_activation))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        # optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


# In[7]:


tuner = kt.Hyperband(model_builder,
     objective='accuracy',  # duda: deberíamos de utilizar 'val_accuracy' (validación) pero no funciona.
     max_epochs=10,
     factor=3,
     directory='hp',
     project_name='cnn_model'
     )


# In[8]:


early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
tuner.search(train_ds, epochs=10, callbacks=[early_stop])


# In[9]:


best_hps= tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
loss, accuracy = best_model.evaluate(val_ds)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[15]:


best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)


# In[14]:

best_model_update = keras.callbacks.ModelCheckpoint('./models/cnn/best', save_best_only=True, monitor='accuracy')
model.fit(train_ds, batch_size=BATCH_SIZE, epochs=50,  callbacks=[early_stop, best_model_update])


# In[ ]:




