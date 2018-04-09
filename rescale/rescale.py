
import numpy as np
import os
import sys
import cv2
from skimage import data, color, io, img_as_float
import matplotlib

def run():
    extension = 'bmp'
    pic = sys.argv[1]
    name = '../colorization/samples/success/' + pic
    small_filename = name + '_out.' + extension
    big_filename = name + '_original.' + extension
    out_filename = name + '_large.' + extension

    big_img = cv2.imread(big_filename)
    height, width = big_img.shape[:2]

    small_img = cv2.imread(small_filename)
    res_img = cv2.resize(small_img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    alpha = 1

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = matplotlib.colors.rgb_to_hsv(big_img)
    color_mask_hsv = matplotlib.colors.rgb_to_hsv(res_img)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = matplotlib.colors.hsv_to_rgb(img_hsv)


    cv2.imwrite(out_filename, img_masked)

if __name__ == '__main__':
    run()
