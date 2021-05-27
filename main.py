from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def images (initial_data):
    processed_data = []
    for line in initial_data:
        for pixel in line:
            if pixel > 255:
                pixel = pixel - 255
            if pixel < 0:
                pixel = 255 - pixel
        image = np.array (line)
        processed_data.append (image)
    return np.array(processed_data)


def model_train(traindata, trainlabels, testdata, testlabels):
    print("model starts training ... \n")
    LR = LogisticRegression(max_iter=1000)

    LR.fit(traindata, trainlabels)
    print("training done ... \n")

    print("model starts predicting ... \n")
    testlabels_predicted = LR.predict(testdata)
    print("prediction done .... \n")

    print(confusion_matrix(testlabels, np.ravel(testlabels_predicted)))

if __name__ == "__main__":
    train_images = np.loadtxt("C:/Users/Iustin/Desktop/data/train_images.txt")
    train_labels = np.loadtxt("C:/Users/Iustin/Desktop/data/train_labels.txt", "float")
    test_images = np.loadtxt("C:/Users/Iustin/Desktop/data/test_images.txt")
    test_labels = np.loadtxt("C:/Users/Iustin/Desktop/data/test_labels.txt", "float")

    train_images_processed = images(train_images)
    test_images_processed = images(test_images)

    model_train(train_images_processed, train_labels, test_images_processed, test_labels)