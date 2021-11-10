import sys
import os.path

import ml
import numpy as np


def usage():
    print("Usage: python3 main.py hw4-1 <N> <mx1> <my1> <mx2> <my2> <vx1> <vy1> <vx2> <vy2>")
    print("Usage: python3 main.py hw4-2")
    exit()


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        usage()

    if (sys.argv[1] == 'hw4-1'):
        if (len(sys.argv) != 11):
            usage()
        ml.Logistic_regression(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(
            sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]))

    elif (sys.argv[1] == 'hw4-2'):
        if (len(sys.argv) != 2):
            usage()
        train_label, train_image, test_label, test_image = ml.read_data(
            './data/')
        ml.EM_algorithm(train_label, train_image, test_label, test_image)

'''
    if (len(sys.argv) < 2):
        usage()

    if (sys.argv[1] == 'hw3-1-a'):
        if (len(sys.argv) != 4):
            usage()
        print(ml.generateUnivariateGaussianData(float(sys.argv[2]), float(sys.argv[3])))
        exit()

    elif (sys.argv[1] == 'hw3-2'):
        if (len(sys.argv) != 4):
            usage()
        ml.sequentialEstimator(float(sys.argv[2]), float(sys.argv[3]))
        exit()

    elif (sys.argv[1] == 'hw3-3'):
        if (len(sys.argv) != 6):
            usage()
        w = np.array(sys.argv[5].split(','), dtype=float)
        ml.bayesianLinearRegression(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), w)

    else:
        usage()
'''
