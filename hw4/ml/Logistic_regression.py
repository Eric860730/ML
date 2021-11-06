import ml
import numpy as np
from scipy.special import expit
import math


def creatDesignMatrix(Data, n):
    # check Data is array or a point.
    if isinstance(Data, np.ndarray):
        row = Data.shape[0]
        design_matrix = np.zeros((row, n), dtype=float)
        for i in range(row):
            for j in range(n):
                design_matrix[i][j] = Data[i][0] ** j
    elif isinstance(Data, int) or isinstance(Data, float):
        design_matrix = np.zeros((1,n), dtype=float)
        for i in range(n):
            design_matrix[0][i] = Data ** i
    return design_matrix

def Find_error_point(w, D1, D2):
    #count the number of D1 and D2 in the lines of left and right.
    num_left_d1 = 0
    num_left_d2 = 0
    num_right_d1 = 0
    num_right_d2 = 0

    # record the index of error point
    l_d1 = -1
    l_d2 = -1
    r_d1 = -1
    r_d2 = -1

    # calculate all points on the side of w
    for i in range(D1.shape[0]):
        #D1
        # xi = [1 xi xi^2]
        xi = creatDesignMatrix(D1[i][0], w.shape[0])
        # yi < L, means (xi, yi) on the left hand side of w.
        if D1[i][1] < np.matmul(xi, w):
            num_left_d1 += 1
            l_d1 = i
        else:
            num_right_d1 += 1
            r_d1 = i

        #D2
        # xi = [1 xi xi^2]
        xi = creatDesignMatrix(D2[i][0], w.shape[0])
        # yi < L, means (xi, yi) on the left hand side of w.
        if D2[i][1] < np.matmul(xi, w):
            num_left_d2 += 1
            l_d2 = i
        else:
            num_right_d2 += 1
            r_d2 = i

    # D1 has more points on the left hand side than D2.
    if num_left_d1 > num_left_d2:
        # choose l_d2 or r_d1 as error point
        if l_d2 != -1:
            return 2, l_d2
        elif r_d1 != -1:
            return 1, r_d1
        # all correct, done.
        else:
            return -1, -1
    # D1 equal D2, choose arbitrary point as error point
    elif num_left_d1 == num_left_d2 and num_left_d1 != 0:
        return 1, l_d1
    elif num_left_d1 < num_left_d2:
        # choose l_d1 or r_d2 as error point
        if l_d1 != -1:
            return 1, l_d1
        elif r_d2 != -1:
            return 2, r_d2
        # all correct, done.
        else:
            return -1, -1



# key formula: Wn+1 = Wn + ∇wJ = Wn + X^T(yi - (1 + e^-Xiw) ^ -1)
# when ∇wJ = 0, it is convergence.
# where X is design matrix, Xi is ith element of X, initial w is n basis of arbitrary elements.
# Finally we get a convergence w, which is a coefficient vector of Logistic regression.
def Gradient_descent(X, Y):
    # suppose basis is 2, given arbitrary w.
    w = np.random.rand(3, 1)
    count = 0
    while True:
        count += 1
        # Calculate ∇wJ = A^T(Yi - (1 + e^-Xiw) ^ -1)
        gradientJ = np.matmul(X.transpose(), Y - np.linalg.inv(1 + math.exp(np.matmul(-X, w))))
        if (abs(gradientJ) < 1e-5).all() or count > 10000:
            return w
        w = w + gradientJ

def Logistic_regression(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
    # Generate D1 and D2
    D1 = np.zeros((n,2))
    D2 = np.zeros((n,2))
    for i in range(n):
        D1[i][0] = ml.generateUnivariateGaussianData(mx1, vx1);
        D1[i][1] = ml.generateUnivariateGaussianData(my1, vy1);
        D2[i][0] = ml.generateUnivariateGaussianData(mx2, vx2);
        D2[i][1] = ml.generateUnivariateGaussianData(my2, vy2);

    # Xw = w1x + w2y + w3, X = [x, y, 1]
    # Y = Bernoulli(f(Xw))
    X = np.zeros((2 * n, 3))
    Y = np.zeros((2 * n, 1))
    X[0:n, 0:2] = D1
    X[n:2 * n, 0:2] = D2
    X[:, 2] = 1
    Y[n:2 * n, 0] = 1
    w = Gradient_descent(X, Y)
    print(w)
    d = ml.generateUnivariateGaussianData(1,1);
    pass
