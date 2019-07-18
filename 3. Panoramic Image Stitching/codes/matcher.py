import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt

from utils import *

# Utils
def SSD(f1, f2):
    f1 = f1.reshape(-1, 1)
    f2 = f2.reshape(-1, 1)
    d = f1-f2
    ssd = np.dot(d.T, d)
    return ssd.item()

class KeypointMatcher:
    def __init__(self, detector='SIFT'):
        if detector == "SIFT":
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif detector == "SURF":
            self.detector = cv2.xfeatures2d.SURF_create()

    def find_best_match(self, f, fs):
        ssds = []
        for f_ in fs:
            ssds.append(SSD(f, f_))
        index_sorted = np.argsort(ssds)
        f1 = fs[index_sorted[0]]
        f2 = fs[index_sorted[1]]
        ratio_distance = SSD(f, f1) / SSD(f, f2)
        f1_index = index_sorted[0]
        return f1_index, ratio_distance
    
    def find_best_matches(self, img1, img2, threshold=0.6, verbose=False):
        if verbose:
            print("Finding Keypoints...", end='\t')
        (keypoints_1, features_1) = self.detector.detectAndCompute(img1, None)
        (keypoints_2, features_2) = self.detector.detectAndCompute(img2, None)
        if verbose:
            print("# keypoints : (%d, %d)" %(len(keypoints_1), len(keypoints_2)))
            print("Finding Matches...", end='\t')
        matches = [] #(index, ratio_dist)
        for f in features_1:
            matches.append(self.find_best_match(f, features_2))
        indices = [match[0] for match in matches]
        dists = [match[1] for match in matches]
        match_points = []
        for i in range(len(matches)):
            if dists[i] < threshold:
                k1 = keypoints_1[i].pt
                k2 = keypoints_2[indices[i]].pt
                k1 = int(k1[0]), int(k1[1])
                k2 = int(k2[0]), int(k2[1])
                match_points.append([k1, k2])
        if verbose:
            print("# Matches: %d"%(len(match_points)))
        return match_points

    def draw_match_points(self, img1, img2, threshold=0.75):
        match_points = self.find_best_matches(img1, img2, threshold)
        img_cat = concatnate_images(img1, img2)
        num_match = len(match_points)
        print("# Matches:", num_match)
        for i, match_point in enumerate(match_points):
            p1, p2 = match_point       
            p2  = p2[0]+img1.shape[1], p2[1]
            color = get_color((i+1)/num_match)
            color = get_color(0.25)
            cv2.line(img_cat, p1, p2, color, 1)
        return img_cat