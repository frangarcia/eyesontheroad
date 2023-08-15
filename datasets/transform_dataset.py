# pip install cmake
# pip install dlib
# pip install argparse

import cv2
import glob2
from imutils import face_utils
import dlib
import imutils

FOLDER_IMAGES_INPUT = './dmd/binary_labels'
FOLDER_IMAGES_OUTPUT = './dmd/binary_labels_faces'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_face(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, width=600)
    faces = detector(img)

    if len(faces) == 0:
        return None

    largest_rect = max(faces, key=lambda rect: rect.area())
    print(f"largest_face: {largest_rect}")
    (x, y, w, h) = (largest_rect.left(), largest_rect.top(), largest_rect.width(), largest_rect.height())

    return img[y:y+h, x:x+w]

file_paths = glob2.glob(FOLDER_IMAGES_INPUT + '/**/*.jpg')

for file_path in file_paths:
    print(f"File input path: {file_path}")
    img = cv2.imread(file_path)
    cv2.imshow('Image', img)
    face_img = detect_face(img)
    if face_img is not None:
        label = 'awake' if 'awake' in file_path else 'drowsy'
        save_path = f"{FOLDER_IMAGES_OUTPUT}/{label}/{file_path.split('/')[-1]}"
        print(save_path)
        cv2.imwrite(save_path, face_img)