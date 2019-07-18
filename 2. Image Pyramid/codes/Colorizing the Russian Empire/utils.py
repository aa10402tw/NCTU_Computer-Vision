import numpy as np


def shift_image(X, sx, sy, set_zero=True):
    X = np.roll(X, sy, axis=0)
    X = np.roll(X, sx, axis=1)
    if set_zero:
        if sy>0:
            X[:sy, :] = 0
        elif sy<0:
            X[sy:, :] = 0
        if sx>0:
            X[:, :sx] = 0
        elif sx<0:
            X[:, sx:] = 0
    return X

# filters
def zero_padding(img, size=(1,1)):
    if type(size) == type(1):
        oy = ox = size
    else:
        oy, ox = size
    m, n = img.shape
    img_pad = np.zeros((m+2*oy, n+2*ox))
    img_pad[oy:-oy, ox:-ox] = img
    return img_pad

def convolution_2D(img, filter, padding=True):
    if len(img.shape) != 2:
        raise Exception("Only support gray image right now !", img.shape)
    m, n = img.shape
    p, q = filter.shape
    
    ox = (q-1)//2
    oy = (p-1)//2
    if padding:
        img = zero_padding(img, size=(oy,ox))
        m, n = img.shape
    
    output_img = np.zeros((m, n))
    for cy in range(oy, m-oy):
        for cx in range(ox, n-ox):
            img_window = img[(cy-oy):(cy+oy)+1:, (cx-ox):(cx+ox)+1]
            output_img[cy, cx] = np.sum(img_window*filter)
            
    output_img = output_img[oy:-oy, ox:-ox]
    return output_img

def get_sobel_filter():
    sobel_x = np.array(
        [[1,0,-1],
         [2,0,-2],
         [1,0,-1]])
    sobel_y = sobel_x.T
    return sobel_x, sobel_y



def subsampling(img, size=(64,64)):
    if len(img.shape) != 2:
        raise Exception("Only support gray image right now !", img.shape)
    m, n = img.shape
    sy = m/size[0]
    sx = n/size[1]
    img_sub = np.zeros(size)
    for row in range(size[0]):
        for col in range(size[1]):
            r = min(row*sy, m-1)
            c = min(col*sx, n-1)
            img_sub[row][col] = img[int(r)][int(c)]
    return img_sub


def crop(image, margin=0.1):
    height, width = image.shape
    y1, y2 = int(margin * height), int((1 - margin) * height)
    x1, x2 = int(margin * width), int((1 - margin) * width)
    return image[y1:y2, x1:x2]

def devide_bgr(img):
    m,n = img.shape
    n_rows = int(m//3)

    img_b = img[:n_rows]
    img_g = img[n_rows:n_rows*2]
    img_r = img[n_rows*2:n_rows*3]
    return img_b, img_g, img_r

def align_bgr(img_b, img_g, img_r, align_vec_g=(0,0), align_vec_r=(0, 0)):
    m, n = img_b.shape
    img_g_shift = shift_image(img_g, align_vec_g[0],  align_vec_g[1])
    img_r_shift = shift_image(img_r, align_vec_r[0],  align_vec_r[1])
    img_color = np.zeros((m,n,3))
    img_color[:,:,0] = img_r_shift
    img_color[:,:,1] = img_g_shift
    img_color[:,:,2] = img_b
    return img_color


def get_similar(r, g, b, rad): 
    bg_similar = similarity(b, g, rad)
    br_similar = similarity(b, r, rad)
    return (bg_similar, br_similar)

def similarity(img_b, img, rad):
    mat = np.zeros((rad*2, rad*2))
    for i in range(-rad, rad):
        for j in range(-rad, rad):
                img_new = np.roll(img, i, axis=0)
                img_new = np.roll(img_new, j, axis=1)
                ssd_val = ssd(img_b, img_new)
                mat[i+rad, j+rad] = ssd_val

    lowest = mat.argmin() 
    row_shift = (lowest // (rad*2)) - rad ##check this part **
    col_shift = (lowest % (rad*2)) - rad ##check this part **
    return (row_shift, col_shift)

def ssd(img_1, img_2):
    ssd = np.sum((img_1 - img_2) **2)
    return ssd