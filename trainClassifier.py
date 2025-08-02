import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataDict = pickle.load(open("./data.pickle", "rb"))


# data was a list, must convert to numpy arrays
data = np.asarray(dataDict["data"])
labels = np.asarray(dataDict["labels"])

# trainging set and testing set

# splitting all the data into 2 sets
xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify = labels)

# create model
model = RandomForestClassifier()

model.fit(xTrain, yTrain)

yPredict = model.predict(xTest)

score = accuracy_score(yPredict, yTest)

print("{}% of samples were classified correctly".format(score * 100))

f = open("model.p", "wb")
pickle.dump({"model":model},f)
f.close()

