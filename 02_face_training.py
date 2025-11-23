import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = "dataset"

# Tentukan folder project (misal folder script ini berada di project root)
project_path = os.path.dirname(os.path.abspath(__file__))

# Folder haarcascades di dalam project
cascade_file = os.path.join(project_path, "haarcascade_frontalface_default.xml")

# Buat recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(cascade_file)

print(f"[INFO] Using cascade file: {cascade_file}")


# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert("L")  # convert it to grayscale
        img_numpy = np.array(PIL_img, "uint8")
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for x, y, w, h in faces:
            faceSamples.append(img_numpy[y : y + h, x : x + w])
            ids.append(id)
    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml

trainer_folder = os.path.join(project_path, "trainer")
os.makedirs(trainer_folder, exist_ok=True)

trainer_file = os.path.join(trainer_folder, "trainer.yml")

recognizer.write(trainer_file)  # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
