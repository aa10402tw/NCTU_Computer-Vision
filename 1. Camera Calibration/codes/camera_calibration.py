
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
np.set_printoptions(suppress=True, precision=4)


def get_normalized_matrix(points):
    N = len(points)
    points = points.reshape(N, -1)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_xs = np.mean(xs)
    mean_ys = np.mean(ys)
    var_xs = np.var(xs)
    var_ys = np.var(ys)

    sx = ((1) / var_xs) ** (1 / 2)
    sy = ((1) / var_ys) ** (1 / 2)

    A = np.array([
        [sx, 0, -sx * mean_xs],
        [0, sy, -sy * mean_ys],
        [0, 0, 1]
    ])
    return A


def get_homography(obj_points, img_points):

    obj_points = np.array([[p[0], p[1], 1] for p in obj_points]).reshape(-1, 3, 1)
    img_points = np.array([[p[0][0], p[0][1], 1] for p in img_points]).reshape(-1, 3, 1)

    N = img_points.shape[0]
    M = np.zeros((2 * N, 9), dtype=np.float64)

    NormMat_obj = get_normalized_matrix(obj_points)
    NormMat_img = get_normalized_matrix(img_points)

    for i in range(N):
        obj_point_norm = np.matmul(NormMat_obj, obj_points[i])
        img_point_norm = np.matmul(NormMat_img, img_points[i])

        X, Y = obj_point_norm[0], obj_point_norm[1]
        u, v = img_point_norm[0], img_point_norm[1]

        row_1 = np.array([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u])
        row_2 = np.array([0, 0, 0, -X, -Y, -1, X * v, Y * v, v])

        M[2 * i] = row_1
        M[2 * i + 1] = row_2

    U, S, V_t = np.linalg.svd(M)
    H_norm_vec = V_t[np.argmin(S)]
    H_norm_mat = H_norm_vec.reshape(3, 3)
    H_mat = np.linalg.inv(NormMat_img).dot(H_norm_mat).dot(NormMat_obj)
    H_mat /= H_mat[2, 2]
    return H_mat


def get_intrinsic_parameters(H_r):
    M = len(H_r)
    V = np.zeros((2 * M, 6), np.float64)

    def v_pq(p, q, H):
        v = np.array([
            H[0, p] * H[0, q],
            H[0, p] * H[1, q] + H[1, p] * H[0, q],
            H[1, p] * H[1, q],
            H[2, p] * H[0, q] + H[0, p] * H[2, q],
            H[2, p] * H[1, q] + H[1, p] * H[2, q],
            H[2, p] * H[2, q]
        ])
        return v
    for i in range(M):
        H = H_r[i]
        V[2 * i] = v_pq(p=0, q=1, H=H)
        V[2 * i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))

    # solve V.b = 0
    U, S, V_t = np.linalg.svd(V)
    b = V_t[np.argmin(S)]

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    # Make sure B is postive defintite
    if(B[0, 0] < 0 or B[1, 1] < 0 or B[2, 2] < 0):
        B = B * -1

    L = np.linalg.cholesky(B)
    K = np.linalg.inv(L).T * L[2, 2]
    return K


def get_extrinsic_parameters(K, Hs):
    K_inv = np.linalg.inv(K)
    extrinsics_pred = []

    for H in Hs:
        h0 = H[:, 0].reshape(3, 1)
        h1 = H[:, 1].reshape(3, 1)
        h2 = H[:, 2].reshape(3, 1)

        lambda_ = 1 / (np.linalg.norm(np.matmul(K_inv, h0)))
        r0 = lambda_ * np.matmul(K_inv, h0)
        r1 = lambda_ * np.matmul(K_inv, h1)
        r2 = np.cross(r0, r1, axis=0)
        r2 /= np.linalg.norm(r2)
        tvec = lambda_ * np.matmul(K_inv, h2)

        if(tvec[-1] < 0):
            tvec *= -1
        Rt = np.hstack((r0, r1, r2, tvec))  # world to model
        extrinsics_pred.append(Rt)

    extrinsics_pred = np.array(extrinsics_pred).reshape(-1, 3, 4)
    return extrinsics_pred


def set_plot_lims(min_values, max_values):
    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

    mid_x = (X_max + X_min) * 0.5
    mid_y = (Y_max + Y_min) * 0.5
    mid_z = (Z_max + Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, 0)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')


###############################
### Collect Correspondences ###
###############################
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x * corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/correct/*.jpg')
# images = glob.glob('data/wrong/*.jpg')
# images = glob.glob('mydata/*.JPG')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
plt.figure(figsize=(10, 10))

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    print('find the chessboard corners of', fname, img.shape)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

##########################
### OpenCV Calibration ###
##########################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1, 6)
extrinsics_opencv = extrinsics
cameraMatrix_opencv = mtx


#######################
### Our Calibration ###
#######################
Hs = []
for (obj_points, img_points) in zip(objpoints, imgpoints):
    H = get_homography(obj_points, img_points)
    Hs.append(H)
K = get_intrinsic_parameters(Hs)
extrinsics_pred = get_extrinsic_parameters(K, Hs)
extrinsics_ours = extrinsics_pred
cameraMatrix_ours = K


###################
### Plot Result ###
###################
# camera parameters
cam_width = 0.064 / 0.1
cam_height = 0.032 / 0.1
scale_focal = 1600

# chess board setting
board_width = 8
board_height = 6
square_size = 1

print('Show the camera extrinsics')
fig = plt.figure(figsize=(15, 5))

# OpenCV Result
ax = fig.add_subplot(1, 2, 1, projection='3d')
plt.title("OpenCV")
camera_matrix = cameraMatrix_opencv.copy()
extrinsics = extrinsics_opencv.copy()
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                 scale_focal, extrinsics, board_width,
                                                 board_height, square_size, True)
set_plot_lims(min_values, max_values)

# Ours Result
ax = fig.add_subplot(1, 2, 2, projection='3d')
camera_matrix = cameraMatrix_ours.copy()
extrinsics = extrinsics_ours.copy()
_, _ = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                               scale_focal, extrinsics, board_width,
                               board_height, square_size, True)
plt.title("Ours")
set_plot_lims(min_values, max_values)


plt.show()


def print_Mat(string, mat):
    print(string)
    for row in mat:
        print('[ ', end='')
        for ele in row:
            print('{:8.3f}'.format(ele), end='\t')
        print(']')

print_Mat('OpenCV Intrinsic : ', cameraMatrix_opencv)
print()
print_Mat('Our Intrinsic : ', K)
