import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import  Dense, Flatten, Conv2D, Rescaling, BatchNormalization, MaxPooling2D, Dropout
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
import random
import cv2
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle

FOLDER_IMAGES = './datasets/dmd/binary_labels'
TEST_FOLDER_IMAGES = './datasets/dmd/binary_labels_test'
INPUT_SHAPE = (180, 180, 3)
BATCH_SIZE = 32
IMG_SIZE = (180, 180)
VAL_SPLIT = 0.2
MODEL_SAVE_FOLDER = './models/cnn/cnn_simple/'
EPOCHS = 30

def crop_face_from_image(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    highlighted_image = image.copy()
    ax = plt.gca()
    y_min = float('inf')
    y_max = float('-inf')
    x_min = float('inf')
    x_max = float('-inf')
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(face_border)
        y_min = min(y_min, y)
        y_max = max(y_max, y + height)
        x_min = min(x_min, x)
        x_max = max(x_max, x + width)
    plt.axis("off")
    plt.show()
    if y_min == float('inf') or y_max == float('-inf') or x_min == float('inf') or x_max == float('-inf'):
        return highlighted_image
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)
    cropped_image = highlighted_image[int(y_min):int(y_max), int(x_min):int(x_max)]
    plt.imshow(cropped_image.astype("uint8"))
    return cropped_image



for image_path in glob(os.path.join(TEST_FOLDER_IMAGES, "*.*")):
    image = plt.imread(image_path)
    crop_face_from_image(image)