import cv2
import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def concatnate_images(img1, img2):
    if len(img1.shape)==3: # color image
        h1, w1, c = img1.shape
        h2, w2, c = img2.shape
        img_cat = np.zeros(( max(h1, h2), w1+w2, c), dtype=img1.dtype)
        img_cat[0:h1, 0:w1] = img1
        img_cat[0:h2, w1:w1+w2] = img2
    return img_cat

def get_color(value, cmap=plt.cm.gist_rainbow):
    color = cmap(value)[:3]
    color = int(color[0]*255), int(color[2]*255), int(color[2]*255)
    return color

def warp(img1, img2, H):
    h1, w1, c = img1.shape
    h2, w2, c = img2.shape
    img_warp = np.zeros( (max(h1,h2), w1+w2, c))
    H_inv = np.linalg.inv(H)
    for x_transform in range(img_warp.shape[1]):
        for y_transform in range(img_warp.shape[0]):
            p_transform = np.array([x_transform, y_transform, 1]).reshape(3,1)
            p = np.matmul(H_inv, p_transform)
            x, y, scale = p.reshape(3)
            x /= scale
            y /= scale
            if (x < img2.shape[1] and x > 0) and (y < img2.shape[0] and y > 0):
                img_warp[y_transform, x_transform] = img2[int(y), int(x)]   
    for x in range(w1):
        for y in range(h1):
            #if np.array_equal(img_warp[y, x], [0,0,0]):
            if True:
                img_warp[y, x] = img1[y, x]
    return img_warp

def crop_black_boundary(img):
    if len(img.shape)==3:
        n_rows, n_cols, c = img.shape
    else:
        n_rows, n_cols = img.shape
    row_low, row_high = 0, n_rows
    col_low, col_high = 0, n_cols
    
    for row in range(n_rows):
        if np.count_nonzero(img[row]) > 0:
            row_low = row
            break
    for row in range(n_rows-1, 0, -1):
        if np.count_nonzero(img[row]) > 0:
            row_high = row
            break
    for col in range(n_cols):
        if np.count_nonzero(img[:, col]) > 0:
            col_low = col
            break
    for col in range(n_cols-1, 0, -1):
        if np.count_nonzero(img[:, col]) > 0:
            col_high = col
            break
    
    return img[row_low:row_high, col_low:col_high]
