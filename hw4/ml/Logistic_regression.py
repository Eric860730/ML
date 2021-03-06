import ml
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, expm1
from scipy.linalg import inv
import math

# python rich for debug
from rich.traceback import install
install(show_locals=True)


def creatDesignMatrix(Data, n):
    # check Data is array or a point.
    if isinstance(Data, np.ndarray):
        row = Data.shape[0]
        design_matrix = np.zeros((row, n), dtype=float)
        for i in range(row):
            for j in range(n):
                design_matrix[i][j] = Data[i][0] ** j
    elif isinstance(Data, int) or isinstance(Data, float):
        design_matrix = np.zeros((1, n), dtype=float)
        for i in range(n):
            design_matrix[0][i] = Data ** i
    return design_matrix


# key formula: Wn+1 = Wn + ∇wJ = Wn + X^T(yi - (1 + e^-Xiw) ^ -1)
# when ∇wJ = 0, it is convergence.
# where X is design matrix, Xi is ith element of X, initial w is n basis of arbitrary elements.
# Finally we get a convergence w, which is a coefficient vector of Logistic regression.
def calculateGradientDescent(X, Y):
    # suppose basis is 2, given arbitrary w.
    w = np.random.rand(3, 1)
    count = 0
    while True:
        count += 1
        old_w = w.copy()
        # Calculate ∇wJ = X^T(Yi - 1 / (1 + e^-Xiw))
        # scipy.special.expit(x) = 1 / (1 + exp(-x))
        gradientJ = np.matmul(X.transpose(), Y - expit(np.matmul(X, w)))
        w = w + gradientJ
        if (np.linalg.norm(w - old_w) < 1e-5) or count > 10000:
            print(f'gradient count {count}')
            return w


# Newton's method: x1 = x0 - f'(x0) / f''(x0) = x0 - Hf(x)^-1 * ∇f(x0), where f(x) = J
# H(J) = X^T * D * X, where Djj = e^-Xijwj/(1+e^-Xijwj)^2 = [1/(1+e^-Xijwj)] * [1 - 1/(1+e^-Xijwj)]
# ∇wJ = X^T(Yi - 1 / (1 + e^-Xiw))
def calculateNewtonRegression(X, Y, n):
    # suppose basis is 2, given arbitrary w.
    w = np.random.rand(3, 1)
    # diagonal matrix D
    D = np.zeros((2*n, 2*n))
    count = 0
    while True:
        count += 1
        old_w = w.copy()

        # calculate diagonal matrix D for Hessian function
        product = X.dot(w)
        diagonal = (expm1(-product) + 1) * np.power(expit(product), 2)
        np.fill_diagonal(D, diagonal)

        # gradientJ = ∇wJ = X^T(Yi - 1 / (1 + e^-Xiw)) = X^T * (Y - logis_f)
        gradientJ = X.transpose().dot(Y - expit(X.dot(w)))

        # Hessian function = X^T * D * X
        Hessian_f = X.transpose().dot(D.dot(X))
        # if Hessian function is none singular, calculate new_w = w + H(J)^-1 * ∇J
        try:
            # Newton, w = w - Hf(x)^-1 * ∇f(x)
            w = w - inv(Hessian_f).dot(X.transpose().dot(Y - expit(X.dot(w))))
        # else,gradientJ use steepest gradient descent
        except:
            w = w + gradientJ
        if (np.linalg.norm(w - old_w) < 1e-5) or count > 10000:
            print(f'Newton count{count}')
            break
    print(f"Newton w: {w}")
    return w


# According our regression result w to classify X
def classifyResult(X, w):
    class1 = []
    class2 = []
    n = X.shape[0] / 2
    # calculate the number of successful classifications s1 and s2
    s1 = 0
    s2 = 0
    for i in range(X.shape[0]):
        if np.matmul(X[i], w) < 0:
            class1.append(X[i, 0:2])
            if i < n:
                s1 += 1
        else:
            class2.append(X[i, 0:2])
            if i >= n:
                s2 += 1
    class1 = np.array(class1)
    class2 = np.array(class2)
    return s1, s2, class1, class2


def visualizeConfusionMatrix(s1, s2, n):
    print(f"Confusion Matrix:")
    print(f"             Predict cluster 1 Predict cluster 2")
    print(f"Is cluster 1        {s1:<2}              {n-s1:<2}")
    print(f"Is cluster 2        {n-s2:<2}              {s2:<2}")
    print(f"\nSensitivity (Successfully predict cluster 1): {s1/n:.5f}")
    print(f"Specificity (Successfully predict cluster 2): {s2/n:.5f}")


def visualizeResult(X, w1, w2, n):
    # draw Ground truth scatter plot
    plt.subplot(131)
    plt.title("Ground truth")
    plt.scatter(X[0:n, 0], X[0:n, 1], c='r')
    plt.scatter(X[n:2*n, 0], X[n:2*n, 1], c='b')

    # Gradient descent
    s1, s2, class1, class2 = classifyResult(X, w1)
    print(
        f"Gradient descent:\n\nw:\n{float(w1[0]):15.10f}\n{float(w1[1]):15.10f}\n{float(w1[2]):15.10f}\n")
    visualizeConfusionMatrix(s1, s2, n)

    # draw Gradient descent
    plt.subplot(132)
    plt.title("Gradient descent")
    plt.scatter(class1[:, 0], class1[:, 1], c='r')
    plt.scatter(class2[:, 0], class2[:, 1], c='b')

    print(f"\n----------------------------------------")
    # Newton's method
    s1, s2, class1, class2 = classifyResult(X, w2)
    print(
        f"Newton's method:\n\nw:\n{float(w2[0]):15.10f}\n{float(w2[1]):15.10f}\n{float(w2[2]):15.10f}\n")
    visualizeConfusionMatrix(s1, s2, n)

    # draw Newton's method
    plt.subplot(133)
    plt.title("Newton's method")
    try:
        plt.scatter(class1[:, 0], class1[:, 1], c='r')
    except:
        _ = 0
    plt.scatter(class2[:, 0], class2[:, 1], c='b')

    plt.show()


def Logistic_regression(n, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2):
    # Generate D1 and D2
    D1 = np.zeros((n, 2))
    D2 = np.zeros((n, 2))
    for i in range(n):
        D1[i][0] = ml.generateUnivariateGaussianData(mx1, vx1)
        D1[i][1] = ml.generateUnivariateGaussianData(my1, vy1)
        D2[i][0] = ml.generateUnivariateGaussianData(mx2, vx2)
        D2[i][1] = ml.generateUnivariateGaussianData(my2, vy2)

    # Xw = w1x + w2y + w3, X = [x, y, 1]
    # Y = Bernoulli(f(Xw))
    X = np.zeros((2 * n, 3))
    Y = np.zeros((2 * n, 1))
    X[0:n, 0:2] = D1
    X[n:2 * n, 0:2] = D2
    X[:, 2] = 1
    Y[n:2 * n, 0] = 1
    w1 = calculateGradientDescent(X, Y)
    w2 = calculateNewtonRegression(X, Y, n)
    visualizeResult(X, w1, w2, n)
