import numpy as np
from libsvm.svmutil import *
import time

# python rich for debug
from rich.traceback import install
install(show_locals=True)


def computeSVMAndTime(kernel_type: int, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
    """
    compute SVM use input kernel_type(kernel function) and compute total cost of time.

    parameters
    kernel_type: type of kernel function, 0 = linear, 1 = polynomial, 2 = radial basis function(RBF)

    <source> https://www.itread01.com/content/1549816773.html
    """
    print('kernel_type: ', end='')
    if kernel_type == 0:
        print('linear')
    elif kernel_type == 1:
        print('polynomial')
    elif kernel_type == 2:
        print('RBF')
    else:
        print('Error kernel type!')
        exit(-1)

    start_time = time.time()
    # -t: choose kernel function, -q: quite mode(no outputs)
    parameters = svm_parameter('-t '+str(kernel_type)+' -q')
    problem  = svm_problem(Y_train, X_train)
    model = svm_train(problem, parameters)
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
    end_time = time.time()
    print(f'total cost {end_time - start_time:.2f} sec.\n')


def SVM(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
    """
    The main function of SVM

    parameters
    X_train: is a 5000x784 matrix. Every row corresponds to a 28x28 grayscale image.
    Y_train: is a 5000x1 matrix, which records the class of the training samples.
    X_test: is a 2500x784 matrix. Every row corresponds to a 28x28 grayscale image.
    Y_test: is a 2500x1 matrix, which records the class of the test samples.
    """
    print('========== Part1 ==========')
    # linear = 0
    computeSVMAndTime(0, X_train, Y_train, X_test, Y_test)
    # polynomial = 1
    computeSVMAndTime(1, X_train, Y_train, X_test, Y_test)
    # radial basis function(RBF) = 2
    computeSVMAndTime(2, X_train, Y_train, X_test, Y_test)


