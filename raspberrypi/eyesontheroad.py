import joblib
import cv2
import numpy as np
import dlib
import imutils
# from matplotlib import pyplot as plt
from imutils import face_utils
from picamera import PiCamera
from scipy.spatial import distance as dist

_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480
_CONSECUTIVE_DROWSY_ALARM = 3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../datasets/dlib/shape_predictor_68_face_landmarks.dat')

def get_landmarks_ratios(frame):
  frame = imutils.resize(frame, width=640)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # detect faces
  faces = detector(gray)
  if len(faces) == 0:
    return None
  # get the largest face
  largest_face = None;
  for face in faces:
    largest_face_area = 0;
    if face.area() > largest_face_area:
      largest_face = face
  shape = predictor(gray, largest_face)
  # Extracting the indices of the facial features
  (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
  (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
  (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
  # Get coordinates for left eye, right eye, and mouth
  left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(lStart, lEnd)]
  right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(rStart, rEnd)]
  mouth = [(shape.part(i).x, shape.part(i).y) for i in range(mStart, mEnd)]
  # Compute aspect ratios for the eyes and mouth
  def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
  def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[0], mouth[4])
    mar = A / C
    return mar
  left_ear = eye_aspect_ratio(left_eye)
  right_ear = eye_aspect_ratio(right_eye)
  mouth_ar = mouth_aspect_ratio(mouth)
  return left_ear, right_ear, mouth_ar

# Load the saved model
model = joblib.load('rfgrid.pkl')

# Load the PiCamera
camera = PiCamera()
camera.resolution = (_IMAGE_WIDTH, _IMAGE_HEIGHT)
consecutive_drowsy = 0
while True:
  img = np.empty((_IMAGE_HEIGHT, _IMAGE_WIDTH, 3), dtype=np.uint8)
  camera.capture(img, 'bgr')
  #cv2.imwrite("/home/pi/image3.jpg", img)

  # img = cv2.imread(file_path)
  ratios = get_landmarks_ratios(img)
  if img is not None and ratios is not None:
    features = np.array([ratios])
    features = features.reshape(1, -1)
    predictions = model.predict(features)
    if predictions[0] == 0:
      consecutive_drowsy += 1
    else:
      consecutive_drowsy = 0
    if consecutive_drowsy > _CONSECUTIVE_DROWSY_ALARM:
      print('\a')
    print('Awake' if predictions[0] else 'Drowsy')
  else:
    print('No faces detected in the image.')








