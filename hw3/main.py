import sys
import os.path

import ml
import numpy as np

def usage():
    print("Usage: python3 main.py hw3-1-a <mean> <variance>")
    print("Usage: python3 main.py hw3-1-b <n> <a> <w>")
    print("Usage: python3 main.py hw3-2 <mean> <variance>")
    print("Usage: python3 main.py hw3-3 <bias int a> <precision int b> <poly_basis int n> <init_prior array w>")
    exit()


if __name__ == '__main__':

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
