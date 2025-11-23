import numpy as np
import cv2
import os


# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

project_path = os.path.dirname(os.path.abspath(__file__))
faceCascade_file = os.path.join(project_path, "haarcascade_frontalface_default.xml")
eyeCascade_file = os.path.join(project_path, "haarcascade_eye.xml")

faceCascade = cv2.CascadeClassifier(faceCascade_file)
eyeCascade = cv2.CascadeClassifier(eyeCascade_file)

cap = cv2.VideoCapture(2)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("video", img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
