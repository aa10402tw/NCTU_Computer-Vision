import numpy as np
import random

def to_homogeneous(points):
    if points.shape[0] == 3:
        return points
    ones = np.ones(points.shape[1])
    homo_points = np.vstack((points, ones))
    return homo_points

def normalize_points(points, img_shape):
    h, w = img_shape[0], img_shape[1]
    ones = np.ones(points.shape[0]).reshape(-1, 1)
    points = np.concatenate((points, ones), axis=1).T
    norm_mat = np.array([[2/w, 0,  -1],
                        [  0, 2/h, -1],
                        [  0,  0,   1]])
    points_norm = np.dot(norm_mat, points)
    return points_norm, norm_mat

def compute_fundamental(x1_sample, x2_sample, T1, T2):
    A = []
    num_points = x1_sample.shape[1]
    for i in range(num_points):
        x1 = x1_sample[:, i]
        x2 = x2_sample[:, i]
        A.append([x1[0]*x2[0], x1[1]*x2[0], x1[2]*x2[0],
                  x1[0]*x2[1], x1[1]*x2[1], x1[2]*x2[1],
                  x1[0]*x2[2], x1[1]*x2[2], x1[2]*x2[2]])

    A = np.array(A)
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F


def RANSAC_fundamental(x1, x2, T1, T2, n_iters=2000, threshold=5e-6, random_seed=None):
    # x: (3, 572)
    best_F = np.zeros((3,3))
    best_num_inlier = 0
    best_mask = []
    if random_seed is not None:
        random.seed( random_seed)

    for iter_ in range(n_iters):
        num_inlier = 0
        idx = random.sample(range(x1[0].shape[0]), 8)
        x1_sample = x1[:, idx]
        x2_sample = x2[:, idx]
        F = compute_fundamental(x1_sample, x2_sample, T1, T2)

        # sampson distance
        Fx1 = np.dot(F.T, x1)
        Fx2 = np.dot(F, x2)
        denom = Fx1[0]**2+Fx1[1]**2+Fx2[0]**2+Fx2[1]**2
        sampson = np.diag( np.dot(x2.T, np.dot(F,x1)) )**2/denom

        mask = []
        # Compute num of inliner
        for j in range (sampson.shape[0]):
            if sampson[j] <= threshold:
                num_inlier += 1
                mask.append(1)
            else:
                mask.append(0)

        if num_inlier > best_num_inlier:
            #Denormalized
            F = np.dot(T2.T, np.dot(F,T1))
            best_F = F
            best_num_inlier = num_inlier
            best_mask = np.array(mask)
    print("My Inliers:", best_num_inlier)

    return best_F/best_F[-1, -1], best_mask, best_num_inlier


def compute_RT_from_essential(E):

    # make sure E is rank 2
    U,S,V = np.linalg.svd(E)
    e = (S[0]+S[1])/2

    E = np.dot(U, np.dot(np.diag([e,e,0]), V))
    
    Z = np.array([[0,1,0], [-1,0,0], [0,0,0]])
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    
    R1 = np.dot(U, np.dot(W,V))
    R2 = np.dot(U, np.dot(W.T,V))

    t1 = np.expand_dims(U[:,2], axis=1)
    t2 = np.expand_dims(-U[:,2], axis=1)

    return t1, t2, R1, R2


def compute_essential(F, K1, K2):
    E = np.dot(K2.T, np.dot(F, K1))
    return E


def compute_epipole(F):
    """
    Compute the epipole from a fundamental matrix F. 
    (Use with F.T for left epipole.)
    """
    
    # return null space of F (Fx=0)
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]