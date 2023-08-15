# pip install opencv-python
# pip install PiCamera
# git clone https://github.com/opencv/opencv.git
import cv2
import time
from picamera import PiCamera

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
fps_counter = 0
start_time = time.time()

cap = cv2.VideoCapture(0)

def detect_face(img):
    #img_array = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/home/pi/cv.jpg', gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imwrite('/home/pi/face.jpg', img)
    return img

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 10

while(True):
    ret, frame = cap.read()

  #camera = PiCamera()
  #camera.resolution = (640, 480)
  #camera.framerate = 24
  frame = BytesIO()
  output = np.empty((480, 640, 3), dtype=np.uint8)
  camera.capture(output, 'bgr')

    # ret, frame = cap.read()

  frame = detect_face(output)

  fps_counter += 1
  elapsed_time = time.time() - start_time
  fps = fps_counter / elapsed_time

    cv2.imshow('frame', frame)

  print(f"FPS: {fps}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# let src = cv.imread('canvasInput');
# let gray = new cv.Mat();
# cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
# let faces = new cv.RectVector();
# let eyes = new cv.RectVector();
# let faceCascade = new cv.CascadeClassifier();
# let eyeCascade = new cv.CascadeClassifier();
# faceCascade.load('haarcascade_frontalface_default.xml');
# let msize = new cv.Size(0, 0);
# faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);
# for (let i = 0; i < faces.size(); ++i) {
# let roiGray = gray.roi(faces.get(i));
# let roiSrc = src.roi(faces.get(i));
# let point1 = new cv.Point(faces.get(i).x, faces.get(i).y);
# let point2 = new cv.Point(faces.get(i).x + faces.get(i).width,
# faces.get(i).y + faces.get(i).height);
# cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
# // detect eyes in face ROI
# eyeCascade.detectMultiScale(roiGray, eyes);
# for (let j = 0; j < eyes.size(); ++j) {
# let point1 = new cv.Point(eyes.get(j).x, eyes.get(j).y);
# let point2 = new cv.Point(eyes.get(j).x + eyes.get(j).width,
# eyes.get(j).y + eyes.get(j).height);
# cv.rectangle(roiSrc, point1, point2, [0, 0, 255, 255]);
# }
# roiGray.delete(); roiSrc.delete();
# }
# cv.imshow('canvasOutput', src);
# src.delete(); gray.delete(); faceCascade.delete();
# eyeCascade.delete(); faces.delete(); eyes.delete();