import numpy as np
import cv2
import pandas as pd

from sfm import *
from plot import *
from matcher import *
from triangulation import *


##################################################
### What you only have to change is right here ###
##################################################

# True for image set 1, False for image set 2
# same_K = True
same_K = False

# True for our data, False for TA's data
# our_data = True
our_data = False

##################################################
##################################################


if same_K:
	if our_data:
		imgName = 'ours'
		img1 = cv2.imread('../data/ours1.jpg')  # left image set 1
		img2 = cv2.imread('../data/ours2.jpg')  # right

		# scale = 1
		# size = (int(img1.shape[1]*scale), int(img1.shape[0]*scale))
		# img1 = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
		# img2 = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)

		# (Best : 1+2) , (2nd : 1+3)
		K = np.array([
			[3499.9607,    0.,     1507.6073],
			[   0.,     3506.6662, 1923.4628],
			[   0.,        0.,        1.    ]
		])

	else:
		imgName = 'Mesona'		
		img1 = cv2.imread('../data/Mesona1.JPG')  # left image set 1
		img2 = cv2.imread('../data/Mesona2.JPG')  # right
		K  = np.array([[1.4219     , 0.0005     , 0.5092],
				   [0          , 1.4219     , 0.3802],
				   [0          , 0          , 0.0010]] )/0.001
	intrinsic_A = K
	intrinsic_B = K
else:
	K1 = np.array([[5426.566895, 0.678017   , 330.096680],
				   [0.000000   , 5423.133301, 648.950012],
				   [0.000000   , 0.000000   , 1.000000  ]])
	R1 = np.array([[0.140626 , 0.989027 ,-0.045273],
				   [0.475766 ,-0.107607 ,-0.872965],
				   [-0.868258, 0.101223 ,-0.485678]] )
	t1 = np.array( [67.479439  ,-6.020049   ,40.224911  ])
	t1 = np.expand_dims(t1, axis=1)

	K2 = np.array([[5426.566895, 0.678017   , 387.430023],
				   [0.000000   , 5423.133301, 620.616699],
				   [0.000000   , 0.000000   , 1.000000  ]])
	R2 = np.array([[ 0.336455 , 0.940689 ,-0.043627],
				   [ 0.446741 ,-0.200225 ,-0.871970],
				   [-0.828988 , 0.273889 ,-0.487611]] )
	t2 = np.array( [62.882744  ,-21.081516  ,40.544052  ])
	t2 = np.expand_dims(t2, axis=1)

	imgName = 'Statue'
	img1 = cv2.imread('../data/Statue1.bmp')  # left image set 1
	img2 = cv2.imread('../data/Statue2.bmp')  # right
	intrinsic_A = K1
	intrinsic_B = K2

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

matcher = KeypointMatcher("SIFT")
points_1, points_2, matches = matcher.find_matches(img1, img2, threshold=0.8)
#pts1, pts2, matches = sift_detector(img1,img2)

pts1 = np.float64(points_1)
pts2 = np.float64(points_2)



##########################
### Fundamental Matrix ###
##########################

# OpenCV Result
F_cv, mask_cv = cv2.findFundamentalMat(pts1,pts2, method=cv2.FM_8POINT + cv2.FM_RANSAC)

# Our Result
norm_pts1, norm_mat1 = normalize_points(pts1, img1.shape)
norm_pts2, norm_mat2 = normalize_points(pts2, img2.shape)
F, mask, best_num_inlier = RANSAC_fundamental(norm_pts1, norm_pts2, norm_mat1, norm_mat2, n_iters=2000, threshold=5e-6, random_seed=100)

print("\nOpenCV Fundamental Matrix : \n",F_cv)
print("\nOur Fundamental Matrix : \n",F)



#####################
### Epipolar Line ###
#####################

# Filter out outlier points
pts1 = pts1[mask.ravel()==1].T
pts2 = pts2[mask.ravel()==1].T
homo_pts1 = to_homogeneous(pts1)
homo_pts2 = to_homogeneous(pts2)
plot_epipolar_line(img1, img2, homo_pts1, homo_pts2, F, F.T, imgName, best_num_inlier, num_lines=best_num_inlier)



#################################
### Check Epipolar Constraint ###
#################################
# print("\nEpipolar Constraint : ")
# for i in range(10):
# 	print(np.dot(homo_pts1[:,i], np.dot(F, homo_pts2[:,i].T)))


########################
### Essential Matrix ###
########################
E = compute_essential(F, intrinsic_A, intrinsic_B)
print("\nE : \n",E)



#########################
### Projection Matrix ###
#########################
if same_K:
	R_A, t_A = np.eye(3), np.zeros((3,1))
	extrinsic_A = np.hstack((R_A, t_A))

	t1, t2, R1, R2 = compute_RT_from_essential(E)
	extrinsic_B =  [
		np.hstack( (R1, t1) ),
		np.hstack( (R1, t2) ),
		np.hstack( (R2, t1) ),
		np.hstack( (R2, t2) )
	]

	P1 = np.dot(intrinsic_A, extrinsic_A)
	P2 = []
	for exB in extrinsic_B:
		P2.append(np.dot(intrinsic_B, exB))
	P2 = np.array(P2)
else:
	t1 = -np.dot(R1, t1)
	t2 = -np.dot(R2, t2)
	extrinsic_A = np.hstack((R1, t1))
	extrinsic_B = np.hstack((R2, t2))
	P1 = np.dot(intrinsic_A, extrinsic_A)
	P2 = np.dot(intrinsic_B, extrinsic_B)


#####################
### Get 3D Points ###
#####################
if same_K:
	X, bestViewIdx = same_K_linear_triangulation(pts1, pts2, R1, R2, t1, t2, P1, P2)
	plot_four_possible_point_cloud(X, imgName)
	X = X[bestViewIdx]
	P2 = P2[bestViewIdx]
	plot_3D_point_cloud(X, E, imgName)
else:
	X = linear_triangulation(pts1, pts2, P1, P2)
	plot_3D_point_cloud(X, E, imgName)



################
### Save CSV ###
################
np.savetxt(("../csv/%s/Cam1_2Dpts_%s.csv" %(imgName, imgName)), pts1.T, delimiter=",")
np.savetxt(("../csv/%s/Cam2_2Dpts_%s.csv" %(imgName, imgName)), pts2.T, delimiter=",")
np.savetxt(("../csv/%s/3Dpts_%s.csv" %(imgName, imgName)), X[:3, :].T, delimiter=",")
np.savetxt(("../csv/%s/Cam1_P_%s.csv" %(imgName, imgName)), P1, delimiter=",")
np.savetxt(("../csv/%s/Cam2_P_%s.csv" %(imgName, imgName)), P2, delimiter=",")
