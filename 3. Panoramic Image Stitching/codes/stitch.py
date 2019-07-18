import numpy as np
import matplotlib.pyplot as plt

from utils import *
from image_pyramid import *
np.set_printoptions(suppress=True)

def get_blending_mask(shape, win_size=30):
    h, w, c = shape
    blending_mask = np.zeros(shape)
    left = max(0, int(w//2-win_size//2))
    right = min(w, int(w//2+win_size//2))
    blending_mask[:, :int(w//2), :] = 1
    blending_mask[:, int(w//2):, :] = 0
    
    for col in range(left, right):
        blending_mask[:, col, :] = 1 - (col-left)/(right-left) 
    return blending_mask
    
def multiband_blend(imgA, imgB, window_size=15):
    gpA, lpA = image_pyramid(imgA)
    gpB, lpB = image_pyramid(imgB)
    
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        alpha = get_blending_mask(la.shape, win_size=window_size)
        ls = la * alpha + lb * (1-alpha)
        LS.append(ls)
     # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        size = (LS[i].shape[1], LS[i].shape[0])
        #ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = pyramid_upsample(ls_, size=size)
        ls_ = cv2.add(ls_, LS[i])
    return ls_

class Stitch():
    def __init__(self, img1, img2, H, fix=0):
        self.img1 = img1
        self.img2 = img2
        h1, w1, c = self.img1.shape
        h2, w2, c = self.img2.shape
        self.resSize = (max(h1,h2), w1+w2, c)

        self.img1_warp = self.get_warp(self.img1, H=None)
        self.img2_warp = self.get_warp(self.img2, H=H)
        
        self.img1_mask = self.get_mask(self.img1_warp)
        self.img2_mask = self.get_mask(self.img2_warp)
        self.fix = fix
    
    def get_warp(self, img, H=None):
        h, w, c = img.shape
        warp_img = np.zeros(self.resSize)

        if np.array_equal(H, None):
            warp_img[:h, :w] = img
        else:
            H_inv = np.linalg.inv(H)
            for x_transform in range(warp_img.shape[1]):
                for y_transform in range(warp_img.shape[0]):
                    p_transform = np.array([x_transform, y_transform, 1]).reshape(3,1)
                    p = np.matmul(H_inv, p_transform)
                    x, y, scale = p.reshape(3)
                    x /= scale
                    y /= scale
                    if (x < img.shape[1] and x > 0) and (y < img.shape[0] and y > 0):
                        warp_img[y_transform, x_transform] = bilinear_interpolate(img, x, y)
        return warp_img/255


    def get_mask(self, img):
        mask = img[not np.array_equal(img, [0,0,0])][0]
        mask = np.array(mask, dtype=bool)
        return mask

    def get_without_blend(self):
        h1, w1, c = self.img1.shape
        h2, w2, c = self.img2.shape
        res_img = np.zeros( (max(h1,h2), w1+w2, c))
        res_img[:h1, :w1] = self.img1/255

        for x_trans in range(w1, res_img.shape[1]):
            for y_trans in range(res_img.shape[0]):
                if (np.array_equal(res_img[y_trans, x_trans], [0,0,0])):
                    res_img[y_trans, x_trans] = self.img2_warp[y_trans, x_trans]

        return crop_black_boundary(res_img*255.0).astype(np.uint8)

    def get_simple_blend(self):
        overlap = (self.img1_mask * 1.0 + self.img2_mask)
        res_img = self.img1_warp + self.img2_warp
        res_img = res_img / np.maximum(overlap, 1)
        return crop_black_boundary(res_img*255.0).astype(np.uint8)
    
    def get_multiband_blend(self, window_size=20):
        for col in range(self.img2_warp.shape[1]):
            if not np.all(self.img2_mask[:, col] == [False, False, False]):
                break
        res_img = self.img1_warp + self.img2_warp
        regionRight = self.img1.shape[1]  # The Right region of the Overlapping Part is the right region of img1
        regionLeft = col  # The Left region of the Overlapping Part is the left region of img2
        regionWidth = regionRight - regionLeft + 1  # Total width of the overlapping 
        overlap_A = self.img1_warp[:, regionLeft:regionRight]
        overlap_B = self.img2_warp[:, regionLeft:regionRight]
        blend = multiband_blend(overlap_A, overlap_B, window_size)
        for col in range(regionLeft, regionRight+1):
            for row in range(self.img2_warp.shape[0]):
                if not (np.array_equal(self.img1_mask[row,col], [False, False, False]) or np.array_equal(self.img2_mask[row,col], [False, False, False])):
                    c = col - regionRight
                    res_img[row,col] = blend[row, c]
        img = np.clip(crop_black_boundary(res_img*255.0), 0, 255).astype(np.uint8)
        return img
        
    def get_weight_blend(self):
        for col in range(self.img2_warp.shape[1]):
            if not np.all(self.img2_mask[:, col] == [False, False, False]):
                break
        res_img = self.img1_warp + self.img2_warp
        regionRight = self.img1.shape[1]  # The Right region of the Overlapping Part is the right region of img1
        regionLeft = col  # The Left region of the Overlapping Part is the left region of img2
        regionWidth = regionRight - regionLeft + 1  # Total width of the overlapping region

        for col in range(regionLeft, regionRight+1):
            for row in range(self.img2_warp.shape[0]):
                alpha = (col - regionLeft) / (regionWidth)
                alpha = 1 - alpha
                # Blending by weight
                if not (np.array_equal(self.img1_mask[row,col], [False, False, False]) or np.array_equal(self.img2_mask[row,col], [False, False, False])):
                    res_img[row,col] = alpha * self.img1_warp[row,col] + (1 - alpha) * self.img2_warp[row,col]
        return crop_black_boundary(res_img*255.0).astype(np.uint8)