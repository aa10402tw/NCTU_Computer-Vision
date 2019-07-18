import numpy as np
np.set_printoptions(suppress=True)

def homography_error(H, p1, p2):
    p1 = np.array([p1[0],p1[1],1]).reshape(3,1)
    p2_estimated = np.matmul(H, p1)
    p2_estimated /= p2_estimated[2]

    p2 = np.array([p2[0],p2[1],1]).reshape(3,1)
    error_vec = p2 - p2_estimated
    return np.linalg.norm(error_vec)


def get_homography(points_A, points_B):
    #loop through correspondences and create assemble matrix
    aList = []
    for p1, p2 in zip(points_A, points_B):
        p1 = np.matrix([p1[0], p1[1], 1])
        p2 = np.matrix([p2[0], p2[1], 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return np.array(h)


# Find Homography from points_1 to points_2
def RANSAC_homography(points_A, points_B, n_sample=1000, sample_size=10, threshold=5, verbose=False):
    if verbose:
        print("Estimate Homography using RANSAC")
    indices = [i for i in range(len(points_A))]
    best_H = np.zeros((3,3))
    best_num_inlier = 0
    for i in range(n_sample):
        # 1. Sample
        np.random.shuffle(indices)
        idx = indices[:sample_size]
        points_A_sampled = points_A[idx]
        points_B_sampled = points_B[idx]
        H = get_homography(points_A_sampled, points_B_sampled)
    
        # 2. Decide inliner & outliner
        num_inlier = 0
        for (pA, pB) in zip(points_A, points_B):
            error = homography_error(H, pA, pB)
            if error < threshold:
                num_inlier += 1
                
        # 3. keep the one has largest inliner
        if num_inlier > best_num_inlier:
            best_num_inlier = num_inlier
            best_H = H

    # 4. Recompute based on the in linear getting from 3.

    points_A_inlier = np.zeros((best_num_inlier, 2))
    points_B_inlier = np.zeros((best_num_inlier, 2))
    cnt = 0
    for (pA, pB) in zip(points_A, points_B):
        error = homography_error(best_H, pA, pB)
        if error < threshold:
            points_A_inlier[cnt] = pA.reshape(-1)
            points_B_inlier[cnt] = pB.reshape(-1)
            cnt += 1
    H = get_homography(points_A_inlier, points_B_inlier)
    if verbose:
        print("Number of inlier:", best_num_inlier)
        print("Homography Matrix\n", H)
    return H