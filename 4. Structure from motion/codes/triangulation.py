import numpy as np

from sfm import *

def linear_triangulation(pts1, pts2, P1, P2):
	pts_3d = np.zeros((4,1)) # Initial (4*1) array, the first col is useless
	for i in range(pts1.shape[1]):
		A = np.array([
				pts1[0,i]*P1[2,:]-P1[0,:],
				pts1[1,i]*P1[2,:]-P1[1,:],
				pts2[0,i]*P2[2,:]-P2[0,:],
				pts2[1,i]*P2[2,:]-P2[1,:]
			])
		U,S,V = np.linalg.svd(A)
		x = np.expand_dims((V[-1] / V[-1,3]), axis=1)
		pts_3d = np.hstack((pts_3d, x))

	return pts_3d[:, 1:]


def same_K_linear_triangulation(pts1, pts2, R1, R2, t1, t2, P1, P2):
	'''
		E : essential matrix 3*3
		x1: match set1  N*2
		x2: match set2  N*2
	return X: 4*N (last row = 1)
	'''
	RT_comb = {
		0 : [R1, t1],
		1 : [R1, t2],
		2 : [R2, t1],
		3 : [R2, t2]
	}

	pts_3d = []
	viewPts_count = []
	for idx, p2 in enumerate(P2):
		tmp = np.zeros((4,1))
		count=0
		for i in range(pts1.shape[1]):
			A = np.array([
					pts1[0,i]*P1[2,:]-P1[0,:],
					pts1[1,i]*P1[2,:]-P1[1,:],
					pts2[0,i]*p2[2,:]-p2[0,:],
					pts2[1,i]*p2[2,:]-p2[1,:]
				])
			U,S,V = np.linalg.svd(A)
			x = np.expand_dims((V[-1] / V[-1,3]), axis=1)

			# Calculate how many points are in front of camera : (X-CamCenter) dot (R[3, :].T)
			res1 = np.dot(x[:3, 0].T, np.array([0,0,1]).T)

			cam2_center = -np.dot(RT_comb[idx][0].T, RT_comb[idx][1])
			res2 = np.dot((x[:3, 0] - cam2_center.T), np.expand_dims(RT_comb[idx][0][-1, :], axis=1))
			if res1 > 0 and res2 > 0:
				count += 1

			tmp = np.hstack((tmp, x))
		viewPts_count.append(count)
		pts_3d.append(tmp[:, 1:])  
	viewPts_count=np.array(viewPts_count)
	print('\nView Points Count :', viewPts_count)
	argmax = np.argmax(viewPts_count)
	pts_3d = np.array(pts_3d)

	return pts_3d, argmax