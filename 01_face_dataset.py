import cv2
import os

# Ensure the dataset folder exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

cam = cv2.VideoCapture(2)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Enter user ID via terminal
face_id = input("Enter user ID: ").strip()

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

count = 0
max_samples = 30  # number of face images to capture

while True:
    ret, img = cam.read()
    if not ret:
        continue  # skip if frame fails to read

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        count += 1
        # Save detected face
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y : y + h, x : x + w])
        print(f"[INFO] Saved image {count}/{max_samples}")

        if count >= max_samples:
            break

    # Display live frame with face rectangle
    cv2.imshow("Face Capture", img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Press ESC to exit
        break
    elif count >= max_samples:
        break

cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Finished capturing faces and cleaned up resources.")
