import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *
from naive_colorize import naive_colorize
from pyramid_colorize import pyramid_colorize


import glob
img_names = glob.glob("datas/*")

for img_name in img_names:
    img = cv2.imread(img_name, -1)
    m, n = img.shape
    img = img / img.max()
    print("Read image (%s), shape : %s"%(img_name, img.shape))

    if m > 5000:
        print("Doing pyramid colorize..")
        img_color = pyramid_colorize(img, show=False)
    else:
        print("Doing naive colorize..")
        img_color = naive_colorize(img, show=False)

    # Save result
    save_name  = 'results/' + img_name.split('datas')[1][1:].split('.')[0] + ".png"
    print("Save result to (%s)\n\n"%save_name)
    img_color = (img_color*255.0*255.0).astype(np.uint16)
    img_bgr = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_name, img_bgr)