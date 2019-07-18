# Image Pyramid
from math import pi, exp
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def get_gaussian_filter():
    gaussian_kernel = np.array(
        [[1, 4, 6, 4,1],
         [4,16,24,16,4],
         [6,24,36,24,6],
         [4,16,24,16,4],
         [1, 4, 6, 4,1]
            ]) / 256
    return gaussian_kernel

def zero_padding(img, size=(1,1)):
    if type(size) == type(1):
        oy = ox = size
    else:
        oy, ox = size
    shape = img.shape
    shape = shape[0] + 2*oy, shape[1] + 2*oy, 3
    img_pad = np.zeros(shape)
    img_pad[oy:-oy, ox:-ox] = img
    return img_pad

def reflect_padding(img, size=(1,1)):
    if type(size) == type(1):
        oy = ox = size
    else:
        oy, ox = size
    shape = img.shape
    shape = shape[0] + 2*oy, shape[1] + 2*oy, 3
    img_pad = np.zeros(shape)
    img_pad[oy:-oy, ox:-ox] = img

    border_row = oy
    border_col = ox
    for row in range(1, oy+1):
        img_pad[border_row-row] = img_pad[border_row+row]
    for col in range(1, ox+1):
        img_pad[:, border_col-col] = img_pad[:, border_col+col]

    border_row = shape[0] - oy -1 
    border_col = shape[1] - ox -1
    for row in range(1, oy+1):
        img_pad[border_row+row] = img_pad[border_row-row]
    for col in range(1, ox+1):
        img_pad[:, border_col+col] = img_pad[:, border_col-col]
    return img_pad

def convolution_2D(img, filter, padding=True):
    m, n, c = img.shape
    p, q = filter.shape
    filter = filter.reshape(p, q, 1)
    ox = (q-1)//2
    oy = (p-1)//2
    if padding:
        img = reflect_padding(img, size=(oy,ox))
        m, n, c = img.shape
    
    output_img = np.zeros((m, n, c))
    for cy in range(oy, m-oy):
        for cx in range(ox, n-ox):
            img_window = img[(cy-oy):(cy+oy)+1:, (cx-ox):(cx+ox)+1]
            output_img[cy, cx] = np.sum(np.sum(img_window*filter, axis=0), axis=0)
    output_img = output_img[oy:-oy, ox:-ox]
    return output_img


def pyramid_dowsample(img):
    gaussian_kernel = get_gaussian_filter()
    img_smooth = convolution_2D(img, gaussian_kernel)
    img_down = img_smooth[::2, ::2]
    return img_down

def pyramid_upsample(img, size=None):
    h, w, c = img.shape
    gaussian_kernel = get_gaussian_filter()
    img_up = np.zeros((h*2, w*2, c)) 
    img_up[::2, ::2] = img
    img_up = convolution_2D(img_up, gaussian_kernel)*4
    if size is not None:
        img_up = cv2.resize(img_up, size)
    return img_up

def image_pyramid(imgA):
    # generate Gaussian pyramid for A
    G = imgA.copy()
    gpA = [G]
    for i in range(6):
        #G = cv2.pyrDown(G)
        G = pyramid_dowsample(G)
        gpA.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        #GE = cv2.pyrUp(gpA[i], dstsize = size)
        GE = pyramid_upsample(gpA[i], size=size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)
    return gpA, lpA