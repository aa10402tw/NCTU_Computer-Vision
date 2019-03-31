import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *

def naive_colorize(img, show=True):
    
    # compute the height of each channel of image (1/3 of total height)
    height = int(np.floor(img.shape[0] / 3.0))
    if show:
        print("img shape", img.shape)
        print('height: ', height)
    b, g, r = devide_bgr(img)

    b = crop(b)
    g = crop(g)
    r = crop(r)
    
    original_b, original_r, original_g = b, r, g

    # Applying sobel filter
    if show:
        print('applying sobel filter...', end=' ')
    sobel_x, sobel_y = get_sobel_filter()
    b = convolution_2D(b, sobel_x)
    b = convolution_2D(b, sobel_y)
    g = convolution_2D(g, sobel_x)
    g = convolution_2D(g, sobel_y)
    r = convolution_2D(r, sobel_x)
    r = convolution_2D(r, sobel_y)
    if show:
        print("finish sobel")
    
    # pyramid implementation 
    window_size = int(height // 5) 

    #compute similarity
    similar_result = get_similar(r, g, b, window_size)
    row_shift_g, col_shift_g = similar_result[0] #x, y shift for g
    row_shift_r, col_shift_r = similar_result[1] #x, y shift for r 

    # create a color image
    if show:
        print('Green Align Vector: (%d, %d)'%(row_shift_g,  col_shift_g))
        print('Red Align Vector:   (%d, %d)'%(row_shift_r,  col_shift_r))
    original_g = shift_image(original_g, col_shift_g, row_shift_g)
    original_r = shift_image(original_r, col_shift_r, row_shift_r)

    img_out = np.dstack([original_r, original_g, original_b])

    # display the image
    if show:
        plt.figure(figsize=(20,20))
        plt.imshow(img_out)
        plt.show()

    return img_out