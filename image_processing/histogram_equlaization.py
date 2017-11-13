import numpy as np
from numpy import array
import cv2
import matplotlib.pyplot as plt
import math
from image_helper import perform_pixelwise_operation

def compare_to_openCV_equalizeHisto(img):
    cv2.imshow('original image', img)
    my_equalized_img = my_equalizeHisto(img)  # use my equalization function
    cv2.imshow('my equalizeHisto', my_equalized_img)
    openCV_equalized_img = cv2.equalizeHist(img)  # use openCV equalization
    cv2.imshow('openCV equalizeHisto', openCV_equalized_img)
    fig = plt.figure()
    fig.subplots_adjust(hspace=1)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.title.set_text('original image histogram')
    ax2.title.set_text('my equalized histogram')
    ax3.title.set_text('openCV histogram')
    ax1.hist(img.ravel(), bins=256)
    ax2.hist(my_equalized_img.ravel(), bins=256)
    ax3.hist(openCV_equalized_img.ravel(), bins=256)
    plt.show()
    cv2.waitKey()


def calc_cumulative_hist(original_hist):
    cumulative_hist = [0] * 256
    cumulative_hist[0] = 255 * original_hist[0]

    for i in range(1, 256):
        cumulative_hist[i] = cumulative_hist[i - 1] + original_hist[i]

    # normalization       X - min(X)
    #             Xn = ----------------
    #                  max(X) - min(X)

    cumulative_hist = (cumulative_hist - np.min(cumulative_hist)) / (np.max(cumulative_hist) - np.min(cumulative_hist))
    return cumulative_hist


def my_equalizeHisto(img):
    hist = np.histogram(img.ravel(), bins=256)[0]
    cumulative_hist = calc_cumulative_hist(hist)
    return perform_pixelwise_operation(img, lambda pix: cumulative_hist[pix])


def bit_plane_slice(pix):
    slicer = int('11000000', 2)
    s = pix[0] & slicer, pix[1] & slicer, pix[2] & slicer
    return s


def plot_hist(img):
    plt.hist(img.ravel(), bins=256)
    plt.show()


def main():
    img1 = cv2.imread('imgs/arctichare.png', 0)
    img2 = cv2.imread('imgs/baboon.png', 0)
    img3 = cv2.imread('imgs/lady.png', 0)
    colored_img = cv2.imread('imgs/baboon.png')

    new_img = perform_pixelwise_operation(img1)                                             # identity transformation
    # new_img = perform_pixelwise_operation(img1, lambda pix: pix*2)                          # linear transformation
    # new_img = perform_pixelwise_operation(img1, lambda pix: 255 - pix)                      # negative
    # new_img = perform_pixelwise_operation(img1, lambda pix: 255*math.log(pix / 255 + 1))    # log transformation
    # new_img = perform_pixelwise_operation(img1, lambda pix: 255* (pix / 255) ** 4)          # power transformation
    # new_img = perform_pixelwise_operation(colored_img, bit_plane_slice)                     # bit plane slice
    # plt.imshow(new_img)  # [..., ::-1]
    # plt.title('bit slice')
    # plt.show()

    # plt.imshow(new_img, cmap='gray')
    # plt.show()


    compare_to_openCV_equalizeHisto(img1)
    compare_to_openCV_equalizeHisto(img2)
    compare_to_openCV_equalizeHisto(img3)


if __name__ == '__main__':
    main()
