import numpy as np
import rawpy
import matplotlib.pyplot as plt


class Negative:
    """ Hold the image and additional data

    Holds image data
    data about the orientation of the negative
    color info
    """

    def __init__(self, image: np.ndarray):
        self.image = image

    def show(self):
        plt.imshow(self.image)
        plt.show()


def from_path(filepath: str) -> Negative:
    with rawpy.imread(filepath) as raw:
        return Negative(raw.postprocess())
