from matplotlib import pyplot as plt
import numpy as np
import cv2

from utils import *
from matcher import *
from stitch import *
from homography import *
from image_pyramid import *


if __name__ == '__main__':
    imgA = cv2.imread('data/1.jpg')
    imgB = cv2.imread('data/2.jpg')

    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

    # Find matching
    matcher = KeypointMatcher()
    match_points = matcher.find_best_matches(imgA, imgB, threshold=0.75, verbose=True)
    points_A = np.array([match_point[0] for match_point in match_points])
    points_B = np.array([match_point[1] for match_point in match_points])

    # Find Homography from points_B to points_A
    H = RANSAC_homography(points_B, points_A,  n_sample=1000, sample_size=4, threshold=5, verbose=True)

    # Show Stitching Result
    stitch = Stitch(imgA, imgB, H)
    img = stitch.get_multiband_blend(window_size=50)
    # Save Result
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    file_name = "result/stitching_img_multiband.jpg"
    cv2.imwrite(file_name, img)
    print("Save result to ", file_name)

        
   