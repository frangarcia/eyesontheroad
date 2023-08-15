import cv2
import imutils
import dlib
import glob2
import matplotlib.pyplot as plt
import sys

DATASET = './dmd/binary_labels_lite'
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_face(frame):
    frame = imutils.resize(frame, width=640)
    cv2.imshow('Frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    print(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        return None

    largest_rect = max(faces, key=lambda rect: rect.area())
    # print(f"largest_face: {largest_rect}")
    # (x, y, w, h) = (largest_rect.left(), largest_rect.top(), largest_rect.width(), largest_rect.height())

    for c in faces:
        x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
        cv2.rectangle(frame, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 1)
        shape = predictor(gray, c)
        for i in range(0, 68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(frame, str(i + 1), (x, y -5), 1, 0.8, (0, 255, 255), 1)
    return frame

file_paths = glob2.glob(DATASET + '/**/*.jpg')

for i, file_path in enumerate(file_paths):
    print(f"File input path: {file_path}")
    img = cv2.imread(file_path)
    img = detect_face(img)
    if img is not None:
        cv2.imshow('Frame', img)
