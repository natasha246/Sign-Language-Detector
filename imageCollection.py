import os
import cv2

# create directory to store images if not already existing
dataDirectory = './data'
if not os.path.exists(dataDirectory):
    os.makedirs(dataDirectory)

# total number of symbols
# number of images for each symbol
numberOfClasses = 36
datasetSize = 100

cap = cv2.VideoCapture(0)

for i in range(numberOfClasses):
    # if the symbol folder does not exist yet, create it
    if not os.path.exists(os.path.join(dataDirectory,str(i))):
        os.makedirs(os.path.join(dataDirectory,str(i)))

    print("Collecting data for class {}".format(i))

    done = False
    while True:
        ret, frame = cap.read()

        # display text on screen
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("frame", frame)

        # check if user presses q
        if cv2.waitKey(25) == ord("q"):
            break

    # capture the images for saving
    counter = 0
    while counter < datasetSize:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(dataDirectory, str(i), "{}.jpg".format(counter)), frame)
        counter +=1

cap.release()
cv2.destroyAllWindows