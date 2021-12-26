import sys
import os.path

import ml
import numpy as np
from PIL import Image


def usage():
    print("Usage: python3 main.py hw6-1 <image number> <gamma_s> <gamma_c> <k(number of cluster)> <method(0: random, 1: k-means++)>")
    print("Usage: python3 main.py hw6-2 <image number> <gamma_s> <gamma_c> <k(number of cluster)> <method(0: random, 1: k-means++)> <cut(0: normalized, 1: ratio)>")
    exit()


def read_image():
    image = Image.open('./data/image%s.png' % sys.argv[2])
    image = np.asarray(image)
    return image


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        usage()

    if (sys.argv[1] == 'hw6-1'):
        if (len(sys.argv) != 7):
            usage()
        # read data
        image = read_image()
        ml.kernelKMeans(
            image, int(
                sys.argv[2]), float(
                sys.argv[3]), float(
                sys.argv[4]), int(
                    sys.argv[5]), int(
                        sys.argv[6]))

    elif (sys.argv[1] == 'hw6-2'):
        if (len(sys.argv) != 8):
            usage()
        # read data
        image = read_image()
        ml.spectral_clustering(
            image, int(
                sys.argv[2]), float(
                sys.argv[3]), float(
                sys.argv[4]), int(
                    sys.argv[5]), int(
                        sys.argv[6]), int(
                            sys.argv[7]))
