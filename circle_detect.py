import os

import cv2
import numpy as np
from PIL import Image


def pipeline(image: cv2.Mat, label: str):
    scratch = cv2.resize(image, (0, 0), fx=1, fy=1)
    gray1 = np.ndarray(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)
    gray2 = np.ndarray(shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8)

    # blurred = cv2.blur(scratch, (3, 3))
    colored = cv2.cvtColor(scratch, cv2.COLOR_BGR2HSV)

    minSat = 128
    minVal = 32
    red_low = np.array((175, minSat, minVal))
    red_high = np.array((180, 255, 255))
    red_low2 = np.array((0, minSat, minVal))
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

    bothR = cv2.GaussianBlur(bothR, (11, 11), 5)
    bothB = cv2.GaussianBlur(gray3, (11, 11), 5)
    # bothB = gray3

    cv2.imshow(f"HSVified {label}", colored)
    cv2.imshow(f"RED {label}", bothR)
    cv2.imshow(f"BLUE {label}", bothB)

    v_third = int(scratch.shape[0] / 3)
    h_third = int(scratch.shape[1] / 3)
    v_fourth = int(scratch.shape[0] / 4)
    v_bot = scratch.shape[0] - v_fourth
    red_left = bothR[v_third:v_bot, :h_third]
    red_center = bothR[v_third:v_bot, h_third:2*h_third]
    red_right = bothR[v_third:v_bot, h_third*2:]
    blue_left = bothB[v_third:v_bot, :h_third]
    blue_center = bothB[v_third:v_bot, h_third:2*h_third]
    blue_right = bothB[v_third:v_bot, h_third*2:]

    red_left_c = cv2.countNonZero(red_left)
    red_center_c = cv2.countNonZero(red_center)
    red_right_c = cv2.countNonZero(red_right)
    blue_left_c = cv2.countNonZero(blue_left)
    blue_center_c = cv2.countNonZero(blue_center)
    blue_right_c = cv2.countNonZero(blue_right)

    left_merge = cv2.cvtColor(red_left, cv2.COLOR_GRAY2BGR)
    left_merge[:, :, 2] = red_left[:, :]
    left_merge[:, :, 1] = 0
    left_merge[:, :, 0] = blue_left[:, :]
    cv2.imshow(f"Left {label}", left_merge)
    center_merge = cv2.cvtColor(red_center, cv2.COLOR_GRAY2BGR)
    center_merge[:, :, 2] = red_center[:, :]
    center_merge[:, :, 1] = 0
    center_merge[:, :, 0] = blue_center[:, :]
    cv2.imshow(f"Center {label}", center_merge)
    right_merge = cv2.cvtColor(red_right, cv2.COLOR_GRAY2BGR)
    right_merge[:, :, 2] = red_right[:, :]
    right_merge[:, :, 1] = 0
    right_merge[:, :, 0] = blue_right[:, :]
    cv2.imshow(f"Right {label}", right_merge)

    print(f"Red : {red_left_c:5d} {red_center_c:5d} {red_right_c:5d}")
    print(f"Blue: {blue_left_c:5d} {blue_center_c:5d} {blue_right_c:5d}")
    # circlesRed = cv2.HoughCircles(bothR, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=40, minRadius=0, maxRadius=0)
    # print(circlesRed)
    # if circlesRed is not None:
    #     circlesRed = np.uint16(np.round(circlesRed))
    #     for i in circlesRed[0, :]:
    #         cv2.circle(image, (i[0], i[1]), i[2], (0, 128, 255), 2)
    #         cv2.circle(image, (i[0], i[1]), 2, (0, 128, 255), 3)
    # else:
    #     print("No RED circles found.")
    #
    # circlesBlue = cv2.HoughCircles(bothB, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=40, minRadius=0, maxRadius=0)
    # print(circlesBlue)
    # if circlesBlue is not None:
    #     circlesBlue = np.uint16(np.round(circlesBlue))
    #     for i in circlesBlue[0, :]:
    #         cv2.circle(image, (i[0], i[1]), i[2], (255, 128, 0), 2)
    #         cv2.circle(image, (i[0], i[1]), 2, (255, 128, 0), 3)
    # else:
    #     print("No BLUE circles found.")
    # cv2.imshow(label, image)


for x in range(21, 22):
    if os.path.exists(f"test{x}.png"):
        path = f"test{x}.png"
    elif os.path.exists(f"test{x}.jpg"):
        path = f"test{x}.jpg"
    else:
        print(f"cannot locate test no. {x}")
        continue
    image = cv2.imread(path)
    TARGET_WIDTH = 720
    pct = TARGET_WIDTH / image.shape[1]
    pipeline(cv2.resize(cv2.imread(path), (0, 0), fx=pct, fy=pct), f"{x}")
cv2.waitKey(0)
cv2.destroyAllWindows()
