import numpy as np
import matplotlib.pyplot as plt
import sys

# python rich for debug
from rich.traceback import install
install(show_locals=True)


def calculateLSE(A, b, lamda):
    # calculate A^TA
    A_T = transposeMatrix(A)
    A_TA = multipleMatrix(A_T, A)

    # calculate lamdaI
    lamdaI = calculateLamdaI(A_TA.shape[0], lamda)

    # calculate A^TA + lamdaI
    A_TAPlusLamdaI = plusMatrix(A_TA, lamdaI)

    # decomposeLU
    L, U = decomposeLU(A_TAPlusLamdaI)
    ATb = multipleMatrix(A_T, b)
    x = inverseLU(L, U, ATb)
    return x


def calculateNewton(A, b):
    # Hf(x) = 2*AT*A
    Hessian_func = 2 * multipleMatrix(transposeMatrix(A), A)
    Hess_L, Hess_U = decomposeLU(Hessian_func)

    # gradient_F = 2*AT*A*x - 2*AT*b
    x = np.random.rand(Hessian_func.shape[0], 1)
    gradient_F = multipleMatrix(Hessian_func, x) - \
        2 * multipleMatrix(transposeMatrix(A), b)

    # x = x0 - HF(x)^-1 * gradient_F
    x = x - inverseLU(Hess_L, Hess_U, gradient_F)
    return x


def plusMatrix(A, B):
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i][j] = A[i][j] + B[i][j]
    return C


def calculateLamdaI(rank, lamda):
    result = np.identity(rank)
    for i in range(rank):
        for j in range(rank):
            if i == j:
                result[i][j] = lamda
    return result


def transposeMatrix(A):
    return A.T


def multipleMatrix(A, B):
    count = A.shape[1]
    rowA = A.shape[0]
    colB = B.shape[1]
    C = np.zeros(shape=(rowA, colB))

    for i in range(rowA):
        for j in range(colB):
            for k in range(count):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C


def decomposeLU(A):
    # init L and U
    L = np.identity(A.shape[0])
    U = A.copy()
    rank = A.shape[0]
    count = rank - 1
    for i in range(rank - 1):
        for j in range(count):
            MultiTime = -(U[i + j + 1][i] / U[i][i])
            U[i + j + 1][:] = U[i + j + 1][:] + MultiTime * U[i][:]
            L[i + j + 1][i] = -MultiTime
        count = count - 1
    return L, U


def inverseLU(L, U, b):
    # Ly = b
    rank = L.shape[0]
    tmp = L.copy()
    count = rank - 1
    for i in range(rank):
        for j in range(count):
            MultiTime = -(tmp[i + j + 1][i] / tmp[i][i])
            tmp[i + j + 1][:] = tmp[i + j + 1][:] + MultiTime * tmp[i][:]
            b[i + j + 1][:] = b[i + j + 1][:] + MultiTime * b[i][:]
        count = count - 1

    # Ux = y
    rank = U.shape[0]
    tmp = U.copy()
    count = rank - 1
    for i in reversed(range(rank)):
        DiaTimes = 1 / tmp[i][i]
        tmp[i][:] = tmp[i][:] * DiaTimes
        b[i][:] = b[i][:] * DiaTimes
        for j in range(count):
            MultiTime = -(tmp[i - j - 1][i] / tmp[i][i])
            tmp[i - j - 1][:] = tmp[i - j - 1][:] + MultiTime * tmp[i][:]
            b[i - j - 1][:] = b[i - j - 1][:] + MultiTime * b[i][:]
        count = count - 1

    return b


def initA_and_b(data, n):
    A = np.ndarray(shape=(data.shape[0], n))
    b = np.ndarray(shape=(data.shape[0], 1))
    for i in range(data.shape[0]):
        b[i] = data[i][1]
        for j in range(A.shape[1]):
            A[i][n - j - 1] = data[i][0]**j
    return A, b


def calculateError(data, x):
    sum = 0
    for i in range(data.shape[0]):
        dist = x[x.shape[0] - 1] - data[i][1]
        for j in range(x.shape[0] - 1):
            dist = dist + x[j] * (data[i][0]**(x.shape[0] - j - 1))
        sum = sum + dist * dist
    print("Totla error:", float(sum))


def printSolution(x):
    for i in range(x.shape[0]):
        if i == x.shape[0] - 1:
            print(float(x[i]))
        else:
            print(str(float(x[i])) + "X^" +
                  str((x.shape[0] - i - 1)) + " + ", end='')


def visualize(data, L_S, N_S):
    lx = np.arange(-6, 8)
    ly = 0
    ny = 0
    for i in range(L_S.shape[0]):
        if i == L_S.shape[0] - 1:
            ly = ly + L_S[i]
            ny = ny + N_S[i]
        else:
            ly = ly + L_S[i] * (lx**(L_S.shape[0] - i - 1))
            ny = ny + N_S[i] * (lx**(N_S.shape[0] - i - 1))

    # plot LSE
    plt.subplot(211)
    plt.plot(lx, ly, linewidth=3)
    plt.scatter(data[:, 0], data[:, 1], color='red')

    # plot Newton
    plt.subplot(212)
    plt.plot(lx, ny, linewidth=3)
    plt.scatter(data[:, 0], data[:, 1], color='red')
    plt.show()


# regularized linear model regression
def regularizedRegression():
    # read csv as np.array
    data = np.genfromtxt(sys.argv[1], delimiter=',')

    # set A and b
    A, b = initA_and_b(data, int(sys.argv[2]))

    LSE_sol = calculateLSE(A, b, sys.argv[3])
    Newton_sol = calculateNewton(A, b)

    # print LSE result.
    print("LSE:\nFitting line: ", end='')
    printSolution(LSE_sol)
    calculateError(data, LSE_sol)

    # print Newton's result
    print("\nNewton's Method:\nFitting line: ", end='')
    printSolution(Newton_sol)
    calculateError(data, Newton_sol)

    visualize(data, LSE_sol, Newton_sol)
