import numpy as np
from numpy import array
import cv2
import matplotlib.pyplot as plt
import math


def perform_pixelwise_operation(img, op=lambda pix: pix * 1):
    img_arr = array(img)
    shape = img_arr.shape
    new_img = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            new_img[x, y] = op(img_arr[x, y])
    return new_img