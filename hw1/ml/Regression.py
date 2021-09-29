import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

# python rich for debug
from rich.traceback import install
install(show_locals=True)

def MatrixPlus(A, B):
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i][j] = A[i][j] + B[i][j]
    return C

def LamdaI(rank, lamda):
    result = np.identity(rank)
    for i in range(rank):
        for j in range(rank):
            if i == j:
                result[i][j] = lamda
    return result

def MatrixTranspose(A):
    return A.T

def MatrixMultiple(A, B):
    count = A.shape[1]
    rowA = A.shape[0]
    colB = B.shape[1]
    C = np.zeros(shape=(rowA, colB))

    for i in range(rowA):
        for j in range(colB):
            for k in range(count):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C

def LUDecomposition(A, L, U):
    rank = A.shape[0]
    count = rank - 1
    for i in range(rank - 1):
        for j in range(count):
            MultiTime = -(U[i+j+1][i] / U[i][i])
            U[i+j+1][:] = U[i+j+1][:] + MultiTime * U[i][:]
            L[i+j+1][i] = -MultiTime
        count = count - 1

def LSE_solveX(A, b, lamda):
    # calculate A^TA
    A_T = MatrixTranspose(A)
    A_TA = MatrixMultiple(A_T, A)

    # calculate lamdaI
    lamdaI = LamdaI(A_TA.shape[0], lamda)

    # calculate A^TA + lamdaI
    A_TAPlusLamdaI = MatrixPlus(A_TA, lamdaI)

    # init L and U
    L = np.identity(A_TAPlusLamdaI.shape[0])
    U = A_TAPlusLamdaI.copy()

    # LUDecomposition
    LUDecomposition(A_TAPlusLamdaI, L, U)

    # b = ATb
    b = MatrixMultiple(A_T, b)

    # Ly = b
    rank = L.shape[0]
    tmp = L.copy()
    count = rank - 1
    for i in range(rank):
        for j in range(count):
            MultiTime = -(tmp[i+j+1][i] / tmp[i][i])
            tmp[i+j+1][:] = tmp[i+j+1][:] + MultiTime * tmp[i][:]
            b[i+j+1][:] = b[i+j+1][:] + MultiTime * b[i][:]
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
            MultiTime = -(tmp[i-j-1][i] / tmp[i][i])
            tmp[i-j-1][:] = tmp[i-j-1][:] + MultiTime * tmp[i][:]
            b[i-j-1][:] = b[i-j-1][:] + MultiTime * b[i][:]
        count = count - 1

    return b



def InverseU(U):
    rank = U.shape[0]
    tmp = U.copy()
    result = np.identity(U.shape[0])
    count = rank - 1
    for i in reversed(range(rank)):
        DiaTimes = 1 / tmp[i][i]
        tmp[i][:] = tmp[i][:] * DiaTimes
        result[i][:] = result[i][:] * DiaTimes
        for j in range(count):
            MultiTime = -(tmp[i-j-1][i] / tmp[i][i])
            tmp[i-j-1][:] = tmp[i-j-1][:] + MultiTime * tmp[i][:]
            result[i-j-1][:] = result[i-j-1][:] + MultiTime * result[i][:]
        count = count - 1
    return result

def InverseL(L):
    rank = L.shape[0]
    tmp = L.copy()
    result = np.identity(L.shape[0])
    count = rank - 1
    for i in range(rank):
        for j in range(count):
            MultiTime = -(tmp[i+j+1][i] / tmp[i][i])
            tmp[i+j+1][:] = tmp[i+j+1][:] + MultiTime * tmp[i][:]
            result[i+j+1][:] = result[i+j+1][:] + MultiTime * result[i][:]
        count = count - 1
    return result

def LSE(A, b, lamda):
    # calculate A^TA
    A_T = MatrixTranspose(A)
    A_TA = MatrixMultiple(A_T, A)

    # calculate lamdaI
    lamdaI = LamdaI(A_TA.shape[0], lamda)

    # calculate A^TA + lamdaI
    A_TAPlusLamdaI = MatrixPlus(A_TA, lamdaI)

    # init L and U
    L = np.identity(A_TAPlusLamdaI.shape[0])
    U = A_TAPlusLamdaI.copy()

    # LUDecomposition
    LUDecomposition(A_TAPlusLamdaI, L, U)

    # calculate the inverse matrix of U and V
    U_Inverse = InverseU(U)
    L_Inverse = InverseL(L)
    print("L_Inverse :\n", L_Inverse, "\n", U_Inverse)

    # A^-1 = U^-1 * L^-1
    InverseA_TA = MatrixMultiple(U_Inverse, L_Inverse)

    # calculate x
    x = MatrixMultiple(InverseA_TA, A_T)
    x = MatrixMultiple(x, b)
    print(x)
    return x


def InitA_b(data, n):
    A = np.ndarray(shape=(data.shape[0], n))
    b = np.ndarray(shape=(data.shape[0], 1))
    for i in range(data.shape[0]):
        b[i] = data[i][1]
        for j in range(A.shape[1]):
            A[i][n-j-1] = data[i][0]**j
    return A, b


def TotalError(data, x):
    sum = 0
    for i in range(data.shape[0]):
        dist = x[x.shape[0] - 1] - data[i][1]
        for j in range(x.shape[0] - 1):
            dist = dist + x[j] * (data[i][0]**(x.shape[0] - j - 1))
        sum = sum + dist * dist
    print("Totla error:",float(sum));

def Print_Solution(x):
    print("Fitting line: ", end = '')
    for i in range(x.shape[0]):
        if i == x.shape[0] - 1:
            print(float(x[i]))
        else:
            print(str(float(x[i]))+"X^"+str((x.shape[0]-i-1))+" + ", end = '')

def myPlot(data, x):
    px = np.arange(-6, 8)
    py = 0
    for i in range(x.shape[0]):
        if i == x.shape[0] - 1:
            py = py + x[i]
        else:
            py = py + x[i] * (px**(x.shape[0]-i-1))
    plt.plot(px, py, linewidth = 3, label = 'LSE')
    plt.scatter(data[:,0], data[:,1], color = 'red')
    plt.show()


# regularized linear model regression
def R_Regression():
    # read csv as np.array
    data = np.genfromtxt(sys.argv[1], delimiter = ',')

    # set A and b
    A, b = InitA_b(data, int(sys.argv[2]))

    x = LSE_solveX(A, b, sys.argv[3])
    Print_Solution(x)
    TotalError(data, x)
    myPlot(data, x)
