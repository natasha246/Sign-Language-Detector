import cv2
import mediapipe as mp
import numpy as np
import pickle


modelDict = pickle.load(open("./model.p", "rb"))
model = modelDict["model"]

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

hands = mpHands.Hands(static_image_mode = True, min_detection_confidence=0.3)

labelsDict = {0:"A", 1:"B", 2:"C", 3:"I", 4:"L", 5:"U"}

while True:

    dataAux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    # convert image to rgb to use mediapipe
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frameRGB)
    # check if a hand is present
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDrawing.draw_landmarks(
                frame,
                handLandmarks,
                mpHands.HAND_CONNECTIONS,
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style()
            )
        for handLandmarks in results.multi_hand_landmarks:
            for i in range (len(handLandmarks.landmark)):
                x = handLandmarks.landmark[i].x
                y = handLandmarks.landmark[i].y
                dataAux.append(x)
                dataAux.append(y)
                x_.append(x)
                y_.append(y)


        x1 = int(min(x_) * W-10)
        y1 = int(min(y_) * H-10)

        x2 = int(max(x_) * W-10)
        y2 = int(max(y_) * H-10)

        prediction = model.predict([np.asarray(dataAux)])

        predictCharacter = labelsDict[int(prediction[0])]


        cv2.rectangle(frame, (x1, y1), (x2,y2), (0,0,0), 4)
        cv2.putText(frame, predictCharacter, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        cv2.waitKey(25)


cap.release()
cv2.destroyAllWindows()