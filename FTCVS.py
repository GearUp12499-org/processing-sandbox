import cv2 as cv
import numpy as np
import sys

# filename = "C:/Users/nishk/test8.jpg"
filename = "test7.jpg"
# C:/Users/nishk/test6.jpg
# C:/Users/nishk/test8.jpg

img = cv.imread(filename)
# print(img.shape) - to check if we are reading the image
cv.imshow("window name", img)
k = cv.waitKey(0)
cv.destroyAllWindows()

# Converts image to HSV
image_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV output", image_HSV)
s = cv.waitKey(0)
cv.destroyAllWindows()

# BGR in the array - lists out what color range to look at
low = np.array([0, 128, 128])
high = np.array([255, 255, 255])

red_mask = cv.inRange(image_HSV, low, high)
red_pixels = red_mask * img[:, :, 2]

cv.imshow("Display Window", red_pixels)
k = cv.waitKey(0)
cv.destroyAllWindows()
