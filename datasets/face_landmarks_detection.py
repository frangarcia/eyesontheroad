# pip install cmake
# pip install dlib
# pip install imutils

import sys
import cv2
import glob2
import dlib
import matplotlib.pyplot as plt
import imutils

FOLDER_IMAGES_INPUT = './dmd/kaggle_ddd'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def detect_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, width=128)
    faces = detector(img)
    print(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        return None

    largest_rect = max(faces, key=lambda rect: rect.area())
    print(f"largest_face: {largest_rect}")
    (x, y, w, h) = (largest_rect.left(), largest_rect.top(), largest_rect.width(), largest_rect.height())

    return img



file_paths = glob2.glob(FOLDER_IMAGES_INPUT + '/**/*.png')

for i, file_path in enumerate(file_paths):
    if i >= 2:
        sys.exit()
    print(f"File input path: {file_path}")
    img = cv2.imread(file_path)
    img = detect_face(img)
    plt.imshow(img)
    plt.show()
