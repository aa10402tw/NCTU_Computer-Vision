# --- Bag-of-Visual-Words --- #

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans

from utils import *

class BoVW:
    def __init__(self, n_words='auto', img_size=(256,256), verbose=True):
        self.n_words = n_words
        self.img_size = img_size
        self.verbose = verbose
        self.timer = Timer()
    
    def sift_feature(self, img):
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, features = sift.detectAndCompute(img, None)
        return features 

    def KM_cluster(self, imgs):
        if self.verbose:
            self.timer.tic()
            print("--Collect all feature vectors...")
        # Collect all feature vectors
        feature_list = []
        for i, img in enumerate(imgs):
            features = self.sift_feature(img)
            if features is None:
                continue
            feature_list.append(features)
        sift_features = np.concatenate(feature_list, axis=0)
        if self.n_words == 'auto':
            self.n_words = int(np.sqrt(sift_features.shape[0]))
        if self.verbose:
            print("Feature Vectors shape:", sift_features.shape)
            print(self.timer.tic())
            print("--Apply K-means Clustering...")
            print("Num of Visual word:", self.n_words)
            
        # Apply K-means Clustering
        model = MiniBatchKMeans(n_clusters = self.n_words)
        model.fit(sift_features)
        self.KM = model
        if self.verbose:
            print(self.timer.tic())
        return model 
    
    def build_histograms(self, imgs):
        if self.verbose:
            print("--Bulid Histograms...")
        histograms = [] 
        for i, img in enumerate(imgs):
            features = self.sift_feature(img)
            if features is None:
                histograms.append([0 for i in range(self.n_words)])
                continue
            clusters = self.KM.predict(features)
            hist, bin_edges = np.histogram(clusters, bins=self.n_words)
            histograms.append(hist)
        if self.verbose:
            print(self.timer.tic())
        return np.array(histograms)