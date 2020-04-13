from __future__ import annotations

import numbers
import os
import time
from pathlib import Path

from virtual_dark import copypasted
from virtual_dark.user_interface import *

import cv2
import imutils as imutils
import matplotlib.pyplot as plt
import numpy as np
import rawpy

from PIL import Image


class Negative:
    """ Hold the image and additional data

    Holds image data
    data about the orientation of the negative
    color info
    """

    def __init__(self, image: np.ndarray, name: str):
        self.name = name
        self.image = image
        self.reduced_w = None
        self.reduced_h = None

    def get_debug_image(self):
        image2show = self.image.copy()
        max_width = 1000

        if np.ma.size(self.image, 0) > max_width:
            image2show = imutils.resize(self.image, width=1000).copy()
            self.reduced_h, self.reduced_w, _ = image2show.shape

        r, g, b = cv2.split(image2show)
        image2show = cv2.merge((b, g, r))

        return image2show

    def show(self, waitKey=True):
        cv2.imshow(self.name, self.get_debug_image())

        if waitKey:
            cv2.waitKey()
            cv2.destroyAllWindows()

    def plot_channel_histogram(self, show=True):
        bins = 255
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.hist(np.ravel(self.image[:, :, 0]), bins=bins)
        ax2.hist(np.ravel(self.image[:, :, 1]), bins=bins)
        ax3.hist(np.ravel(self.image[:, :, 2]), bins=bins)

        if show:
            plt.show()

    def get_resize_ratios(self):
        h, w, _ = self.image.shape
        return h / self.reduced_h, w / self.reduced_w

    def save_to_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        im = Image.fromarray(np.clip(self.image, 0, 255))
        im.save(os.path.join(dir, self.name + ".tiff"))

    def find_holes(self):
        """ Find the holes in the filmstrip
        TODO: Should probably clean this up a bit more, kind of a lot of stuff here
        """
        threshold_val = 220
        resized = imutils.resize(self.image, width=1000)
        self.reduced_h, self.reduced_w, _ = resized.shape

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

    def rotate_degrees(self, rot_deg):
        rotated = copypasted.rotate_image(self.image, rot_deg)
        return Negative(rotated, self.name)

    def rotate_according_to_slope(self, slope) -> Negative:
        if slope > 1 or slope < -1:
            raise ValueError("Slope has invalid value: {}".format(slope))
        rot_deg = np.rad2deg(np.arcsin(slope))
        rotated = copypasted.rotate_image(self.image, rot_deg)
        return Negative(rotated, self.name + "-rotated")

    def calc_film_white_point(self, hole_centers) -> (int, int, int):
        pts = []

        for i in range(len(hole_centers) - 1):
            p1 = hole_centers[i]
            p2 = hole_centers[i + 1]
            pts.append(get_halfway_point(p1, p2))

        r_h, r_w = self.get_resize_ratios()
        pts = [(int(y * r_h), int(x * r_w)) for y, x in pts]

        sw = 5  # Sample width/2, 5 is pretty okay

        pixels = [self.image[y - sw:y + sw, x - sw:x + sw, :] for x, y in pts]
        pixels = np.stack(pixels)
        n, r, c, _ = pixels.shape

        pixels = pixels.reshape(n * r * c, 3)

        r, g, b = zip(*pixels)

        r_m = np.median(r)
        g_m = np.median(g)
        b_m = np.median(b)
        # r_m = np.mean(r)
        # g_m = np.mean(g)
        # b_m = np.mean(b)

        white_point = r_m, g_m, b_m

        # white_point = (171.5, 62.0, 26.0) # A pretty good val
        return white_point

    def correct_with_white_point(self, whiteRGB: (int, int, int)) -> Negative:
        w, h, _ = self.image.shape
        shape = (w, h)

        lum = sum(whiteRGB) / 3
        multR = lum / whiteRGB[0]
        multG = lum / whiteRGB[1]
        multB = lum / whiteRGB[2]

        r_chan = np.full(shape, multR, dtype=np.float16)
        g_chan = np.full(shape, multG, dtype=np.float16)
        b_chan = np.full(shape, multB, dtype=np.float16)

        mult_array = np.dstack((r_chan, g_chan, b_chan))
        del r_chan, g_chan, b_chan

        corrected = np.multiply(self.image, mult_array).clip(0, 255).astype(np.uint8)

        return Negative(corrected, self.name + "-color_balanced")

    Num = numbers.Number
    NumRange = (Num, Num)
    Rect = (int, int, int, int)  # x, y, w, h

    def get_color_ranges_in_area(self, area: Rect) -> (NumRange, NumRange, NumRange):
        x, y, w, h = area
        image_area = self.image[x:x + w, y:y + h, :]

        red_range = (image_area[:, :, 0].min(), image_area[:, :, 0].max())
        green_range = (image_area[:, :, 1].min(), image_area[:, :, 1].max())
        blue_range = (image_area[:, :, 2].min(), image_area[:, :, 2].max())

        print("Red range: {}\nGreen range: {}\nBlue range: {}".format(red_range, green_range, blue_range))

        return red_range, green_range, blue_range

    def correct_color_curves(self, red_range: NumRange, green_range: NumRange, blue_range: NumRange) -> Negative:
        img_range = (0, 255)

        red = maprange(self.image[:, :, 0], red_range, img_range).astype(np.uint8)
        green = maprange(self.image[:, :, 1], green_range, img_range).astype(np.uint8)
        blue = maprange(self.image[:, :, 2], blue_range, img_range).astype(np.uint8)

        corrected = np.dstack((red, green, blue)).astype(np.uint8)

        return Negative(corrected, self.name + "-corrected")

    def change_channel_level(self, channels, mult) -> Negative:
        image = self.image.copy()
        for i in channels:
            image[:, :, i] = np.clip((mult * image[:, :, i]).astype(np.uint8), 0, 255).astype(np.uint8)
        return Negative(image, self.name + "-level_changed")

    def invert(self) -> Negative:
        inverted = 255 - self.image

        return Negative(inverted, self.name + "-inverted")


def from_path(filepath: str) -> Negative:
    with rawpy.imread(filepath) as raw:
        name = Path(filepath).stem
        return Negative(raw.postprocess(), name)


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


def get_halfway_point(p1, p2):
    x = p1[0] + (p2[0] - p1[0]) / 2
    y = p1[1] + (p2[1] - p1[1]) / 2
    return x, y


def maprange(s, a, b):
    (a1, a2), (b1, b2) = a, b
    return b1 + (s - a1) * ((b2 - b1) / (a2 - a1))


def fully_process_neg(negative) -> Negative:
    centers = negative.find_holes()

    #    white_point = negative.calc_film_white_point(centers)
    #    print("white_point:", white_point)

    slope, _ = fit_line_through_holes(centers)
    if slope < -1 or slope > 1:
        rot90 = negative.rotate_degrees(-90)
        centers90 = rot90.find_holes()
        slope, _ = fit_line_through_holes(centers90)
        rotated = rot90.rotate_according_to_slope(slope)
    else:
        rotated = negative.rotate_according_to_slope(slope)
    del negative

    debug_im = rotated.get_debug_image()
    rect = (1200, 850, 2300, 1850)
    x, y, w, h = rect
    hr, wr = rotated.get_resize_ratios()
    resized_rect = int(x / wr), int(y / hr), int(w / wr), int(h / hr)
    p1, p2 = debug_draw_rect(debug_im, resized_rect)
    x, y = p1
    w, h = p2[0]-x, p2[1]-y
    rect = int(x*wr), int(y*hr), int(w*wr), int(h*hr)
    rr, gr, br = rotated.get_color_ranges_in_area(rect)
    color_corrected = rotated.correct_color_curves(rr, gr, br)
    del rotated

    inverted = color_corrected.invert()
    del color_corrected

    return inverted


