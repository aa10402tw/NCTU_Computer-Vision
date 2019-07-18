
import glob
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import time

# Calculate time elapse
class Timer:
    def __init__(self):
        self.time = time.time()
    
    def tic(self):
        time_elapse = time.time() - self.time
        self.time = time.time()
        return "({} min, {} sec)\n".format(int(time_elapse)//60, int(time_elapse)%60)

# Data Conversion
ALL_CATEGORIES = os.listdir("data/train/")
num_classes = len(ALL_CATEGORIES)
ALL_LABELS = [i for i in range(len(ALL_CATEGORIES))]
CAT2LABEL = dict(zip(ALL_CATEGORIES, ALL_LABELS))
LABEL2CAT = dict(zip(ALL_LABELS, ALL_CATEGORIES))
def cat2label(category):
    return CAT2LABEL[category]

def label2cat(label):
    return LABEL2CAT[label]

def compute_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

def get_data(df):
    imgs = []
    labels = []
    img_paths = df['img_paths']
    img_labels = df['img_labels']
    for (img_path, img_label) in zip(img_paths, img_labels):
        img = cv2.imread(img_path, 0)
        imgs.append(img)
        labels.append(img_label)
    return imgs, labels



def plot_confusion_matrix(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    num_instance = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0.0%'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    #cm /= cm_sum
    cm = np.divide(cm,cm_sum)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual Labels'
    cm.columns.name = 'Predicted Labels'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues")
    plt.show()