import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from utils import *

def tiny_img(img, size=(16, 16)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

if __name__ == '__main__':
    # Prepare Data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    n_train_data = len(train_df)
    imgs_train, labels_train = get_data(train_df)
    tiny_imgs_train = list(map(tiny_img, imgs_train))
    X_train = np.array(tiny_imgs_train).reshape((n_train_data, -1))
    y_train = np.array(labels_train)

    n_test_data = len(test_df)
    imgs_test, labels_test = get_data(test_df)
    tiny_imgs_test = list(map(tiny_img, imgs_test))
    X_test = np.array(tiny_imgs_test).reshape((n_test_data, -1))
    y_test = np.array(labels_test)

    from sklearn.neighbors import KNeighborsClassifier
    print("{:^3} | {:^10} | {:^10}".format("K", "Train Acc", "Test Acc"))
    print('-'*27)
    for k in range(1, 30+1):

        # ----  Change Here ---- #
        model = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=1)
        model.fit(X_train, y_train) 

        y_pred = model.predict(X_train)
        train_acc = compute_accuracy(y_pred, y_train)
        
        y_pred = model.predict(X_test)
        test_acc = compute_accuracy(y_pred, y_test)
        # ----  Change Here ---- #

        print("{:^3} | {:^10} | {:^10}".format(k, "%.4f"%train_acc, "%.4f"%test_acc))