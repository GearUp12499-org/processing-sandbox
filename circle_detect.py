import cv2
import numpy as np
from PIL import Image


def pipeline(image: cv2.Mat):
    scratch = cv2.resize(image, (0, 0), fx=1, fy=1)
    cv2.imshow("reference", scratch)
    gray1 = np.ndarray(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
    gray2 = np.ndarray(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

    blurred = cv2.blur(scratch, (3, 3))
    colored = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    red_low = np.array((180 - 5, 128, 64))
    red_high = np.array((180, 255, 255))
    red_low2 = np.array((0, 128, 64))
    red_high2 = np.array((5, 255, 255))

    gray1 = cv2.inRange(colored, red_low, red_high)
    gray2 = cv2.inRange(colored, red_low2, red_high2)

    bothR = cv2.bitwise_or(gray1, gray2)
    # bothR = cv2.blur(bothR, (5, 5))
    # bothR = cv2.threshold(bothR, 254, 255, cv2.THRESH_BINARY)[1]

    blue_low = np.array((190 / 2, 128, 64))
    blue_high = np.array((250 / 2, 255, 255))

    gray3 = cv2.inRange(colored, blue_low, blue_high)
    both = cv2.bitwise_or(gray3, bothR)
    red_sc = cv2.bitwise_and(scratch, scratch, mask=both[:, :])
    cv2.imshow("piped", red_sc)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


pipeline(cv2.imread("cone_left.jpg"))
