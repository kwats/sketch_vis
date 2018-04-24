import numpy as np
import cv2
import argparse
import imutils
from PIL import Image, ImageEnhance

import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])

# Prepocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
flag, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Find contours
img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Select long perimeters only
perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
numcards=len(listindex)

card_number = -1 #just so happened that this is the worst case
stencil = np.zeros(img.shape).astype(img.dtype)
cv2.drawContours(stencil, [contours[listindex[card_number]]], 0, (255, 255, 255), cv2.FILLED)
res = cv2.bitwise_and(img, stencil)

img = imutils.resize(img, width=400)
v = np.median(img)

canny = cv2.Canny(res, 200, 500)
canny = imutils.resize(canny, width=400)


cv2.imwrite("canny.png", canny)
cv2.imwrite("temp.png", img)
paint = cv2.imread("original_paint.png")
paint = imutils.resize(paint, width=400)
img2 = cv2.imread("canny.png")
#img2 = cv2.bitwise_not(img2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
paint = cv2.cvtColor(paint, cv2.COLOR_BGR2HSV)

arr = np.asarray(paint)
h = paint[..., 0]  # All Red values
s = paint[..., 1]  # All Green values
v = paint[..., 2]  # All Blue values

arr2 = np.asarray(img2)
h2 = img2[..., 0]  # All Red values
s2 = img2[..., 1]  # All Green values
v2 = img2[..., 2]  # All Blue values

#h, s, v = cv2.split(paint)
#h2, s2, v2 = cv2.split(img2)
for i, pixel in enumerate(v):
        v2[i] += 1
        v[i] = v[i]  - (v2[i]/7)


out = cv2.merge([h, s, v])
out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
cv2.imwrite('out.png', out)
