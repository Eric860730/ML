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


def printGridSearchResult(best_accuracy: float, best_parameter: dict):
    print(f'Best performing accuracy: {best_accuracy:.2f}%')
    print(f'Best parameters: {best_parameter}\n')

def gridSearch(kernel_type: int, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
    """
    Use grid search method to find the best performing parameters in each kernel function.

    If '-v' is specified in 'options' (i.e., cross validation)either accuracy (ACC) or mean-squared error (MSE) is returned.
        -v n: n-fold cross validation mode

    -t kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v, the dot product of u and v
        -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    1 -- polynomial: (gamma*u'*v + coef0)^degree
        -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        -g gamma : set gamma in kernel function (default 1/num_features)
        -r coef0 : set coef0 in kernel function (default 0), coef0 is the intercept of polynomial.
        -d degree : set degree in kernel function (default 3)
    2 -- radial basis function: exp(-gamma*|u-v|^2)
        -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        -g gamma : set gamma in kernel function (default 1/num_features)
    p.s. the means of cost and gamma
        The larger the gamma value, the smaller the range of influence of data point.
        The larger the cost(C) value, the smaller the tolerance of outlier.
    """
    # set n fold
    num_n = 3

    best_accuracy = 0
    best_parameter = []
    if kernel_type == 0:
        print('kernel function: linear')
        # set parameters
        cost = [np.power(10.0, i) for i in range(-1, 2)]

        for c in cost:
            parameters = svm_parameter('-t '+str(kernel_type)+' -c '+str(c)+' -v '+str(num_n)+' -q')
            problem = svm_problem(Y_train, X_train)
            accuracy = svm_train(problem, parameters)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameter = {'cost': c}

    elif kernel_type == 1:
        print('kernel function: polynomial')
        # set parameters
        cost = [np.power(10.0, i) for i in range(-1, 2)]
        gamma = [1.0 / 784] + [np.power(10.0, i) for i in range(-1, 1)]
        coef0 = [np.power(10.0, i) for i in range(-1, 2)]
        degree = [i for i in range(0, 4)]

        for c in cost:
            for g in gamma:
                for r in coef0:
                    for d in degree:
                        parameters = svm_parameter('-t '+str(kernel_type)+' -c '+str(c)+' -g '+str(g)+' -r '+str(r)+' -d '+str(d)+' -v '+str(num_n)+' -q')
                        problem = svm_problem(Y_train, X_train)
                        accuracy = svm_train(problem, parameters)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_parameter = {'cost': c, 'gamma': g, 'coef0': r, 'degree': d}

    elif kernel_type == 2:
        print('kernel function: RBF')
        # set parameters
        cost = [np.power(10.0, i) for i in range(-1, 2)]
        gamma = [1.0 / 784] + [np.power(10.0, i) for i in range(-1, 1)]

        for c in cost:
            for g in gamma:
                parameters = svm_parameter('-t '+str(kernel_type)+' -c '+str(c)+' -g '+str(g)+' -v '+str(num_n)+' -q')
                problem = svm_problem(Y_train, X_train)
                accuracy = svm_train(problem, parameters)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_parameter = {'cost': c, 'gamma': g}

    else:
        print('Invalid kernel_type[0 ~ 2], exit')
        exit(-1)

    # print best parameter and accuracy
    printGridSearchResult(best_accuracy, best_parameter)


def SVM(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
    """
    The main function of SVM

    parameters
    X_train: is a 5000x784 matrix. Every row corresponds to a 28x28 grayscale image.
    Y_train: is a 5000x1 matrix, which records the class of the training samples.
    X_test: is a 2500x784 matrix. Every row corresponds to a 28x28 grayscale image.
    Y_test: is a 2500x1 matrix, which records the class of the test samples.
    """
    print('===== Compare the performance of linear, polynomial, RBF kernel function =====')
    # linear = 0
    #computeSVMAndTime(0, X_train, Y_train, X_test, Y_test)
    # polynomial = 1
    #computeSVMAndTime(1, X_train, Y_train, X_test, Y_test)
    # radial basis function(RBF) = 2
    #computeSVMAndTime(2, X_train, Y_train, X_test, Y_test)

    print('\n===== Use grid search method to find the best performing parameters in each kernel function =====')
    # linear = 0
    gridSearch(0, X_train, Y_train, X_test, Y_test)
    # polynomial = 1
    gridSearch(1, X_train, Y_train, X_test, Y_test)
    # radial basis function(RBF) = 2
    gridSearch(2, X_train, Y_train, X_test, Y_test)


