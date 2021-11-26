import sys
import os.path

import ml
import numpy as np


def usage():
    print("Usage: python3 main.py hw5-1 (automatically read input data in path 'data/input.data')")
    print("Usage: python3 main.py hw4-2")
    exit()


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        usage()

    if (sys.argv[1] == 'hw5-1'):
        if (len(sys.argv) != 2):
            usage()
        input_data = np.genfromtxt('./data/input.data', delimiter=' ')
        ml.gaussianProcessRegression(input_data)

    elif (sys.argv[1] == 'hw4-2'):
        if (len(sys.argv) != 2):
            usage()
        train_label, train_image, test_label, test_image = ml.read_data(
            './data/')
        ml.EM_algorithm(train_label, train_image, test_label, test_image)
