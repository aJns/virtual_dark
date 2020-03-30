from time import sleep

import imutils as imutils
import numpy as np
import rawpy
import matplotlib.pyplot as plt
import cv2


class Negative:
    """ Hold the image and additional data

    Holds image data
    data about the orientation of the negative
    color info
    """

    def __init__(self, image: np.ndarray):
        self.image = image

    def show(self):
        image2show = self.image
        max_width = 1000

        if np.ma.size(self.image, 0) > max_width:
            image2show = imutils.resize(self.image, width=1000)

        cv2.imshow("Image", image2show)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def find_holes(self):
        """ Find the holes in the filmstrip
        TODO: Should probably clean this up a bit more, kind of a lot of stuff here
        """
        threshold_val = 220
        resized = imutils.resize(self.image, width=1000)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (5, 5))
        thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_im = resized.copy()
        cv2.drawContours(contour_im, contours, -1, (0, 255, 0), 3)

        contour_areas = np.array([cv2.contourArea(c) for c in contours])

        max_dist = 100
        median = np.median(contour_areas)
        filter_list = np.array([np.abs(ca - median) < max_dist for ca in contour_areas])

        contours = np.array(contours)

        centers = map(get_contour_center, contours[filter_list])
        centers = sorted(centers, key=lambda x: x[0])

        return centers

    def rotate_according_to_slope(self, slope):
        rot_deg = np.rad2deg(np.arcsin(slope))
        rotated = imutils.rotate(self.image, rot_deg)
        return Negative(rotated)

    def correct_with_white_point(self, white_point):
        return self


def from_path(filepath: str) -> Negative:
    with rawpy.imread(filepath) as raw:
        return Negative(raw.postprocess())


def get_contour_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def fit_line_through_holes(holes: [(float, float)]) -> (float, float):
    cX, cY = map(list, zip(*holes))
    cX = np.array(cX)
    slope, offset = np.polyfit(cX, cY, 1)
    return slope, offset


def calc_film_white_point(hole_centers) -> (int, int, int):
    return 255, 255, 255


def fully_process_neg(negative) -> Negative:
    centers = negative.find_holes()
    white_point = calc_film_white_point(centers)
    slope, _ = fit_line_through_holes(centers)
    rotated = negative.rotate_according_to_slope(slope)
    color_corrected = rotated.correct_with_white_point(white_point)
    return color_corrected


def white_balance_correction(whiteRGB: (int, int, int), pixel: np.array) -> np.array:
    lum = sum(whiteRGB) / 3
    imgR = pixel[0] * lum / whiteRGB[0]
    imgG = pixel[1] * lum / whiteRGB[1]
    imgB = pixel[2] * lum / whiteRGB[2]
    return np.array([imgR, imgG, imgB])
