import sys
import ml
import os.path


def usage():
    print("Usage: python3 main.py hw3-1-a <mean> <variance>")
    print("Usage: python3 main.py hw3-1-b <n> <a> <w>")
    print("Usage: python3 main.py hw3-2 <mean> <variance>")
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
        ml.bayesianLinearRegression()

    else:
        usage()
