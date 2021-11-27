import sys
import os.path

import ml
import numpy as np


def usage():
    print("Usage: python3 main.py hw5-1 (automatically read input data in path 'data/input.data')")
    print("Usage: python3 main.py hw5-2")
    exit()


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        usage()

    if (sys.argv[1] == 'hw5-1'):
        if (len(sys.argv) != 2):
            usage()
        # read data
        input_data = np.genfromtxt('./data/input.data', delimiter=' ')
        ml.gaussianProcessRegression(input_data)

    elif (sys.argv[1] == 'hw5-2'):
        if (len(sys.argv) != 2):
            usage()
        # read data
        X_train = np.genfromtxt('./data/X_train.csv', delimiter=',')
        Y_train = np.genfromtxt('./data/Y_train.csv', delimiter=',')
        X_test = np.genfromtxt('./data/X_test.csv', delimiter=',')
        Y_test = np.genfromtxt('./data/Y_test.csv', delimiter=',')
        ml.SVM(X_train, Y_train, X_test, Y_test)
