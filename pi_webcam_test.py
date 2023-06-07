import joblib
import cv2
import time
import numpy as np
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
from sklearn.preprocessing import MinMaxScaler
import beepy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./datasets/dlib/shape_predictor_68_face_landmarks.dat')
model = joblib.load('./models/landmarks/rfgrid.pkl')


def get_landmarks_ratios(frame):

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

    #print the face rectangle
    x_ini, y_ini, x_fin, y_fin = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()
    cv2.rectangle(frame, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 1)

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

    return [left_ear, right_ear, mouth_ar], frame


import cv2

cap = cv2.VideoCapture(0)
frame_count = 0
start_time = time.time()
font_size = 0.5

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    frame_count += 1
    frame_height = frame.shape[0]

    output = get_landmarks_ratios(frame)
    if output is not None:
        ratios, frame = output
        features = np.array(ratios)
        features = features.reshape(1, -1)
        predictions = model.predict(features)
        status = 'Awake' if predictions[0] else 'Drowsy'

        cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ratiosr = [round(ratio, 2) for ratio in ratios]
        cv2.putText(frame, f'Ratios: {ratiosr}', (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)


    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)


    cv2.imshow('Webcam Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
