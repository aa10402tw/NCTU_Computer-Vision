import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_relative_keypoints(img1, img2, pts_A, pts_B, num_points=30):
	"""
	Function to plot matching points
	pts_A, pts_B are matrices of keypoints.
	"""
	f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,10))
	for idx in range(num_points):
		x = pts_A[0, idx]
		y = pts_A[1, idx]
		ax1.scatter(x,y, marker='x')
	for idx in range(num_points):
		x = pts_B[0, idx]
		y = pts_B[1, idx]
		ax2.scatter(x,y, marker='x')
	ax1.axis('off'), ax1.imshow(img1)
	ax2.axis('off'), ax2.imshow(img2)
	plt.show()
	

def plot_four_possible_point_cloud(Xs, imgName):
	fig = plt.figure()
	fig.suptitle('Four possible 3D Reconstruction', fontsize=16)

	for idx, X in enumerate(Xs):
		ax = fig.add_subplot(2,2,idx+1, projection='3d')
		ax.plot(X[0,:], X[1,:], X[2,:], 'b.')
		ax.set_xlabel('x axis')
		ax.set_ylabel('y axis')
		ax.set_zlabel('z axis')
	plt.savefig(("../res/PossibleP_%s.png") %(imgName))
	plt.show()

def plot_3D_point_cloud(X, E, imgName):
	fig = plt.figure()
	fig.suptitle('3D reconstructed Result', fontsize=16)
	ax = fig.add_subplot(111, projection='3d')

	ax.plot(X[0,:], X[1,:], X[2,:], 'b.')
	ax.set_xlabel('x axis')
	ax.set_ylabel('y axis')
	ax.set_zlabel('z axis')
	plt.savefig(("../res/FinalRes_%s.png") %(imgName))
	plt.show()


def plot_epipolar_line(img1, img2, x1, x2, F1, F2, imgName, best_num_inlier, num_lines=30):
	"""
	Plot the epipole and epipolar line F*x=0 in an image.
	F is the fundamental matrix,
	x is a point in the other image.
	"""
	if num_lines > best_num_inlier:
		num_lines = best_num_inlier

	def draw_lines(img, ax, pts, x, y, title):
		ax.axis('off'), ax.imshow(img), ax.set_title(title)
		for idx in range(num_lines):
			y_ = y[:, idx]
			crop = np.where((y_>=0) & (y_<img.shape[0]))
			y_ = y_[crop]
			x_ = x[crop]
			ax.plot(x_, y_, linewidth=1)
			ax.scatter(pts[0, idx], pts[1, idx], marker='x')

	def draw_points(img, ax, pts, title):
		ax.axis('off'), ax.imshow(img), ax.set_title(title)
		for idx in range(num_lines):
			ax.scatter(pts[0,idx], pts[1,idx], marker='x')

	m,n = img1.shape[:2]
	line1 = np.dot(F2,x2[:, :num_lines])
	line2 = np.dot(F1,x1[:, :num_lines])
	
	x = np.linspace(0,n,100)
	
	# au + bv + c = 0
	# v = (-au-c)/b
	y1 = np.array([(line1[0, :]*xx+line1[2, :])/(-line1[1, :]) for xx in x])
	y2 = np.array([(line2[0, :]*xx+line2[2, :])/(-line2[1, :]) for xx in x])
	
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10)) #((15,10), (10,8), (10,15))
	draw_points(img1, ax1, x1, title='Left Image')
	draw_lines(img2, ax2, x2, x, y2, title='Right Image')
	draw_lines(img1, ax3, x1, x, y1, title='Left Image')
	draw_points(img2, ax4, x2, title='Right Image')
	plt.savefig(("../res/EpiploarLine_%s.png") %(imgName))
	plt.show()