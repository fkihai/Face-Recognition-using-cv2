import cv2
import numpy as np
import os
import requests
import time

project_path = os.path.dirname(os.path.abspath(__file__))
faceCascade_file = os.path.join(project_path, "haarcascade_frontalface_default.xml")
trainer_file = os.path.join(project_path, "trainer", "trainer.yml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_file)
faceCascade = cv2.CascadeClassifier(faceCascade_file)

font = cv2.FONT_HERSHEY_SIMPLEX
names = ["None", "Fikri", "Delia"]

cam = cv2.VideoCapture(2)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

post_url = "http://localhost:5678/webhook/detection"

# Dictionary untuk menyimpan timestamp terakhir pengiriman tiap ID
last_sent_time = {}
send_interval = 5  # detik, minimal jeda antar POST request per ID

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id_num, confidence = recognizer.predict(gray[y : y + h, x : x + w])

        if confidence < 80:
            id_text = names[id_num]
            conf_text = "{0}%".format(round(100 - confidence))
        else:
            id_text = "unknown"
            conf_text = "{0}%".format(round(100 - confidence))

        cv2.putText(img, str(id_text), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(conf_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # --- SEND TO POST REQUEST WITH THROTTLE ---
        now = time.time()
        if id_text not in last_sent_time or (
            now - last_sent_time[id_text] > send_interval
        ):
            _, img_encoded = cv2.imencode(".jpg", img)
            try:
                files = {"image": ("face.jpg", img_encoded.tobytes(), "image/jpeg")}
                data = {"id": id_text, "confidence": conf_text}
                r = requests.post(post_url, files=files, data=data)
                print(f"[INFO] Sent POST for {id_text}. Status code: {r.status_code}")
                last_sent_time[id_text] = now
            except Exception as e:
                print(f"[ERROR] Failed to send POST request: {e}")

    cv2.imshow("camera", img)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:  # ESC to exit
        break

cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Exiting program and cleanup done.")
