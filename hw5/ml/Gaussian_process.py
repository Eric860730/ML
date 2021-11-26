import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# python rich for debug
from rich.traceback import install
install(show_locals=True)

def rationalQuadraticKernel(Xi: np.ndarray, Xj: np.ndarray, length = 1.0, alpha = 1.0):
    """
    parameter
    Xi: the array of all X
    Xj: the array of all X
    length: The bigger length^2 is, the less wiggly your random functions are, float > 0, default = 1.0
    alpha: weighting of the individual length scales, float > 0, dafault = 1.0
    <source>: https://stats.stackexchange.com/questions/407254/intuition-behind-the-length-scale-of-the-rational-quadratic-kernel

    Rational Quadratic Kernel
    k(Xi, Xj) = (1 + dist(xi, xj)^2 / 2*alpha*length^2)^(-alpha)
    , where dist(xi, xj) is the Euclidean distance of xi and xj.
    """
    kernel_function = (1 + cdist(Xi, Xj, 'sqeuclidean') / (2 * alpha * (length ** 2))) ** (-alpha)
    return kernel_function

def computeGaussianProcess(X: np.ndarray, Y: np.ndarray, X_star: np.ndarray, beta: float, length = 1.0, alpha = 1.0):
    """
    parameter
    X and Y: Input data points
    X_star: All point X we want to predict. In this implement, the boundary of X_star is [-60, 60].
    beta: The inverse of variance of ϵ.
    length: The bigger length^2 is, the less wiggly your random functions are, float > 0, default = 1.0
    alpha: weighting of the individual length scales, float > 0, dafault = 1.0

    Gaussian Process Regression
    yn = f(Xn) + ϵn, ϵn ~ N(.|0, β^-1)
    for f = [f(X1), ..., f(XN)]^T and y = [y1, ..., yN]^T
    p(y|f) = N(y|f, β^-1*I), p(f) = N(0, K)
    marginal likelihood p(y) = ∫p(y|f)p(f)df = N(y|0, C)
    where C(Xn, Xm) = k(Xn, Xm) + β^-1ξnm, which ξnm means when n = m , then ξnm = 1; otherwise, ξnm = 0. i.e. ξnm = identity matrix of n

    Prediction
    1. calculate k(x, x*), k(x, x*)^T, k(x*, x*) + β^-1
    2. conditional
    µ(x*) = k(x, x*)^T * C^-1 * y
    σ^2(x) = k* - k(x, x*)^T * C^-1 * k(x, x*), where k* = k(x*, x*) + β^-1
    """
    # Gaussian process regression
    # k(x, x)
    k_x_x = rationalQuadraticKernel(X, X, length, alpha)
    # covariance matrix
    C = k_x_x + (1 / beta) * np.identity(X.shape[0])

    # Prediction
    # k(x, x*)
    k_x_xStar = rationalQuadraticKernel(X, X_star, length, alpha)
    # k(x*, x*)
    k_xStar_xStar = rationalQuadraticKernel(X_star, X_star, length, alpha)
    # k*
    k_star = k_xStar_xStar + 1 / beta
    # µ(x*)
    predict_mean_xStar = np.matmul(np.transpose(k_x_xStar), np.matmul(np.linalg.inv(C), Y))
    # σ^2(x)
    predict_variance_xStar = k_star - np.matmul(np.transpose(k_x_xStar), np.matmul(np.linalg.inv(C), k_x_xStar))
    return predict_mean_xStar, predict_variance_xStar


def visualizeGaussianProcess(X: np.ndarray, Y: np.ndarray, X_star: np.ndarray, predict_mean_xStar: np.ndarray, predict_variance_xStar: np.ndarray):
    # Draw Gaussian Process Regression
    plt.subplot(211)
    plt.title('Gaussian Process Regression')
    plt.scatter(X, Y, s = 10.0)
    var = np.diagonal(predict_variance_xStar).copy().reshape(-1, 1)
    drawRegression(X_star, predict_mean_xStar, var)
    plt.savefig('foo.png')


def drawRegression(x, y, var):
    plt.plot(x, y, linewidth = 0.5, color = 'black')
    plt.plot(x, y + var, linewidth = 0.5, color = 'red')
    plt.plot(x, y - var, linewidth = 0.5, color = 'red')
    plt.xlim(-60.0, 60.0)
    plt.ylim(-3.0, 3.0)

def gaussianProcessRegression(input_data: np.ndarray):
    """
    parameter
    input_data: The numpy array with all points.
    """
    # training data points (X, Y)
    X = input_data[:,0].reshape(-1, 1)
    Y = input_data[:,1].reshape(-1, 1)

    # create the array of X_star, reshape(-1, 1) means reshape array to one column and automatically compute the number of rows.
    num_slice = 1200
    X_star = np.linspace(-60, 60, num_slice).reshape(-1, 1)

    # set β = 5.0
    beta = 5.0

    # compute Gaussian Process
    # default length and alpha are 1.0
    length = 1.0
    alpha = 1.0
    predict_mean_xStar, predict_variance_xStar = computeGaussianProcess(X, Y, X_star, beta, length, alpha)
    visualizeGaussianProcess(X, Y, X_star, predict_mean_xStar, predict_variance_xStar)
