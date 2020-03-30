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
        threshold_val = 220
        resized = imutils.resize(self.image, width=1000)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (5, 5))
        thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_im = resized.copy()
        cv2.drawContours(contour_im, contours, -1, (0, 255, 0), 3)

        #        contours = imutils.grab_contours(contours)

        contour_areas = np.array([cv2.contourArea(c) for c in contours])
        #        plt.hist(contour_areas)
        #        plt.show()
        max_dist = 100
        median = np.median(contour_areas)
        filter_list = np.array([np.abs(ca - median) < max_dist for ca in contour_areas])
        #        print(contour_areas)
        #        print(contour_areas[filter_list])

        contours = np.array(contours)

        centers = map(get_contour_center, contours[filter_list])
        centers = sorted(centers, key=lambda x: x[0])

        cX, cY = map(list, zip(*centers))
        cX = np.array(cX)
        b, m = np.polyfit(cX, cY, 1)

        rot_deg = np.rad2deg(np.arcsin(b))
        print("Angle: ", rot_deg)

#        print("line: ", b + m * cX)
#        print("cX: ", cX, "b: ", b, "m: ", m)

        #        plt.plot(cX, cY, '.')
        #        plt.plot(cX, m+b*cX, '-')
        #        plt.show()

        fitted_line = [(int(x), int(m + b * x)) for x in cX]

        for c in centers:
            cX, cY = c
            # draw the contour and center of the shape on the image
            cv2.circle(contour_im, (cX, cY), 7, (255, 0, 255), -1)
            cv2.putText(contour_im, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        p1 = fitted_line[0]
        p2 = fitted_line[-1]
        cv2.line(contour_im, p1, p2, (0, 0, 255), 2)

        rotated = imutils.rotate(contour_im, rot_deg)

        cv2.imshow("Threshold image", thresh)
        cv2.imshow("Contours", contour_im)
        cv2.imshow("Rotated", rotated) # Works, fuck yeah
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#        plt.imshow(resized)
#        plt.show()


def from_path(filepath: str) -> Negative:
    with rawpy.imread(filepath) as raw:
        return Negative(raw.postprocess())


def get_contour_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY
