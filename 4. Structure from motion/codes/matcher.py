import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt

from utils import *
    
class KeypointMatcher:
    def __init__(self, detector='SIFT'):
        if detector == "SIFT":
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif detector == "SURF":
            self.detector = cv2.xfeatures2d.SURF_create()

    def find_matches(self, img1, img2, threshold=0.8):
        # find the keypoints and descriptors with detector
        keypoint1, feature1 = self.detector.detectAndCompute(img1, None)
        keypoint2, freaute2 = self.detector.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
    
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(feature1, freaute2, k=2)
        
        good_match = []
        points_1 = []
        points_2 = []
    
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < threshold * n.distance:
                good_match.append(m)
                points_2.append(keypoint2[m.trainIdx].pt)
                points_1.append(keypoint1[m.queryIdx].pt)

        return points_1, points_2, good_match  
