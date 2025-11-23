import cv2
import os
import time

# Ensure the dataset folder exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

cam = cv2.VideoCapture(2)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_id = input("Enter user ID: ").strip()
print("\n[INFO] Initializing face capture. Look at the camera and wait...")

count = 0
max_samples = 30  # number of face images to capture
last_save_time = 0  # timestamp terakhir menyimpan
save_interval = 0.5  # detik

while True:
    ret, img = cam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        now = time.time()
        if now - last_save_time >= save_interval:
            count += 1
            cv2.imwrite(
                f"dataset/User.{face_id}.{count}.jpg", gray[y : y + h, x : x + w]
            )
            print(f"[INFO] Saved image {count}/{max_samples}")
            last_save_time = now

            if count >= max_samples:
                break

    cv2.imshow("Face Capture", img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or count >= max_samples:
        break

cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Finished capturing faces and cleaned up resources.")
