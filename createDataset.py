import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

hands = mpHands.Hands(static_image_mode = True, min_detection_confidence=0.3)


dataDirectory = "./data"

data = []
labels = []

for dir in os.listdir(dataDirectory):
    
    # make sure to skip invalid directories
    dirPath = os.path.join(dataDirectory, dir)
    if not os.path.isdir(dirPath):
        continue

    # iterate through images
    for imagePath in os.listdir(os.path.join(dataDirectory,dir)):
        dataAux = []
        image = cv2.imread(os.path.join(dataDirectory,dir,imagePath))

        # convert image to rgb to use mediapipe
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(imageRGB)
        # check if a hand is present
        if results.multi_hand_landmarks:
            # iterate through each image
            for handLandmarks in results.multi_hand_landmarks:
                for i in range (len(handLandmarks.landmark)):
                    x = handLandmarks.landmark[i].x
                    y = handLandmarks.landmark[i].y
                    dataAux.append(x)
                    dataAux.append(y)

            data.append(dataAux)
            labels.append(dir)

f = open("data.pickle", "wb")
pickle.dump({"data":data, "labels":labels},f)
f.close()