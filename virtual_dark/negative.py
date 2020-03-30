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
        cv2.imshow("image", self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def find_holes(self):
        """ Find the holes in the filmstrip """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (15, 15))
        thresh = blurred
#        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_im = self.image.copy()
        cv2.drawContours(contour_im, contours, -1, (0, 255, 0), 3)

        resized = imutils.resize(contour_im, width=1000)
        cv2.imshow("image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#        plt.imshow(resized)
#        plt.show()


def from_path(filepath: str) -> Negative:
    with rawpy.imread(filepath) as raw:
        return Negative(raw.postprocess())
