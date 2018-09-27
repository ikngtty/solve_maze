import functools

import cv2 as cv
import numpy as np

from mylib import imgcvt, maze, util


def no_change(input):
    return input


def binarize(input):
    return cv.adaptiveThreshold(src=input,
                                maxValue=255,
                                adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                                thresholdType=cv.THRESH_BINARY,
                                blockSize=101,
                                C=10)


def denoise(input):
    return cv.medianBlur(input,
                         ksize=35)


def compress(input):
    ratio = 25
    compressed_shape = tuple(i // ratio for i in input.shape)
    compressed_size = tuple(reversed(compressed_shape))
    return cv.resize(src=input,
                     dsize=compressed_size)


def binarize2(input):
    _, img = cv.threshold(src=input,
                          thresh=150,
                          maxval=255,
                          type=cv.THRESH_BINARY)
    return img


def check_point(input, point):
    """
    To see where a certain point in the image, put a gray square at the point.

    To be exact, the square's top left corner is the one.
    """
    square = [point + util.Point(dy, dx)
              for dy in range(7)
              for dx in range(7)]
    output = np.ndarray.copy(input)
    for p in square:
        output[p] = 150
    return output


def draw_path_fromto(input, start_point, goal_point):
    path = maze.get_path(input, start_point, goal_point)
    # HACK: Implemented roughly.
    delta_square = [util.Point(y, x)
                    for y in range(3)
                    for x in range(3)]
    spread_path = [p + d
                   for p in path
                   for d in delta_square]
    output = np.ndarray.copy(input)
    for p in spread_path:
        output[p] = 150
    return output


def paint_path(input):
    output = np.ndarray(shape=input.shape + (3,))
    for iy, row in enumerate(input):
        for ix, grayscale in enumerate(row):
            # When visualizing cost for debugging.
            # if 147 < grayscale and grayscale < 153:
            #     output[iy, ix] = [0, 0, 255]
            # else:
            #     output[iy, ix] = [grayscale, grayscale, grayscale]
            if grayscale > 200:
                output[iy, ix] = [255, 255, 255]
            elif grayscale < 50:
                output[iy, ix] = [0, 0, 0]
            else:
                output[iy, ix] = [40, 70, 110]
    return output


if __name__ == "__main__":
    start_point = util.Point(22, 70)
    goal_point = util.Point(55, 80)

    check_start_point = functools.partial(check_point, point=start_point)
    check_goal_point = functools.partial(check_point, point=goal_point)
    draw_path = functools.partial(draw_path_fromto,
                                  start_point=start_point,
                                  goal_point=goal_point)

    GRAY = imgcvt.ImageConvert.IMREAD_GRAYSCALE
    COLOR = imgcvt.ImageConvert.IMREAD_COLOR

    converts = (
        imgcvt.ImageConvert("白黒化", no_change, GRAY),
        imgcvt.ImageConvert("二値化", binarize, GRAY),
        imgcvt.ImageConvert("ノイズ除去", denoise, GRAY),
        imgcvt.ImageConvert("画質縮小", compress, GRAY),
        imgcvt.ImageConvert("二値化2", binarize2, GRAY),
        # imgcvt.ImageConvert("スタート地点確認", check_start_point, GRAY),
        # imgcvt.ImageConvert("ゴール地点確認", check_goal_point, GRAY),
        imgcvt.ImageConvert("経路描画", draw_path, GRAY),
        imgcvt.ImageConvert("経路着色", paint_path, GRAY)
    )
    runner = imgcvt.ImageConvertRunner("resources/maze.jpg", converts)
    runner.run_all()

    print("Finish converting your image !!")
