import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sklearn

from utils import *
from bovw import BoVW

if __name__ == '__main__':
    # Prepare Data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    bovw = BoVW(n_words=300, img_size=(512,512))

    imgs_train, labels_train = get_data(train_df)
    bovw.KM_cluster(imgs_train)
    X_train = bovw.build_histograms(imgs_train)
    y_train = np.array(labels_train)

    # Test Data
    imgs_test, labels_test = get_data(test_df)
    X_test = bovw.build_histograms(imgs_test)
    y_test = np.array(labels_test)

    print("--Data Shape")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    
    print("{:^3} | {:^10} | {:^10}".format("K", "Train Acc", "Test Acc"))
    print('-'*27)
    for k in range(1, 30+1):

        # ----  Change Here ---- #
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=1)
        model.fit(X_train, y_train) 

        y_pred = model.predict(X_train)
        train_acc = compute_accuracy(y_pred, y_train)
        
        y_pred = model.predict(X_test)
        test_acc = compute_accuracy(y_pred, y_test)
        # ----  Change Here ---- #

        print("{:^3} | {:^10} | {:^10}".format(k, "%.4f"%train_acc, "%.4f"%test_acc))
    plot_confusion_matrix(y_pred, y_test, labels= [i for i in range(num_classes)])