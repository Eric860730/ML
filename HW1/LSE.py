import numpy as np
from rich.traceback import install
install(show_locals=True)

def MatrixPlus(A, B):
    C = np.zeros(A.shape)
    for i in range(A.ndim):
        for j in range(A.ndim):
            C[i][j] = A[i][j] + B[i][j]
    return C

def LamdaI(rank, lamda):
    result = np.identity(rank)
    for i in range(rank):
        for j in range(rank):
            if i == j:
                result[i][j] = lamda
    return result



def MatrixTransport(A):
    return A.T

def MatrixMultiple(A, B):
    return np.matmul(A, B)

def LUDecomposition(A, L, U):
    rank = A.shape[0]
    count = rank - 1
    for i in range(rank - 1):
        for j in range(count):
            MultiTime = -(U[i+j+1][i] / U[i][i])
            U[i+j+1][:] = U[i+j+1][:] + MultiTime * U[i][:]
            L[i+j+1][i] = -MultiTime
        count = count - 1
    print("L = ", L, "\n U = ", U)


def InverseU(U):
    rank = U.shape[0]
    result = U.copy()
    count = rank - 1
    for i in reversed(range(rank - 1)):
        for j in range(count):
            MultiTime = -(result[i-j-1][i] / result[i][i])
            result[i-j-1][:] = result[i-j-1][:] + MultiTime * result[i][:]
        count = count - 1
    return result

def InverseL(L):
    rank = L.shape[0]
    result = L.copy()
    count = rank - 1
    for i in range(rank - 1):
        for j in range(count):
            MultiTime = -(result[i+j+1][i] / result[i][i])
            result[i+j+1][:] = result[i+j+1][:] + MultiTime * result[i][:]
        count = count - 1
    return result


# read csv as np.array
data = np.genfromtxt('test.csv', delimiter = ',')

# set A and b
A = data.copy()
print(A.shape[0])
b = np.ndarray(shape=(A.shape[0],1))
for i in range(data.shape[0]):
    b[i] = A[i][1]
    A[i][1] = 1

# calculate A^TA
A_T = MatrixTransport(A)
A_TA = MatrixMultiple(A_T, A)
print(A_TA)

# calculate lamdaI
lamdaI = LamdaI(A_TA.shape[0], 0)

# calculate A^TA + lamdaI
A_TAPlusLamdaI = MatrixPlus(A_TA, lamdaI)

# init L and U
L = np.identity(A_TAPlusLamdaI.shape[0])
U = A_TAPlusLamdaI.copy()

# LUDecomposition
LUDecomposition(A_TAPlusLamdaI, L, U)

# calculate the inverse matrix of U and V
InverseU = InverseU(U)
InverseL = InverseL(L)

# A^-1 = U^-1 * L^-1
InverseA_TA = MatrixMultiple(InverseU, InverseL)

# calculate x
x = MatrixMultiple(InverseA_TA, A_T)
x = MatrixMultiple(x, b)
print("===========================")
print(x)
