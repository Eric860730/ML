import numpy as np
from ml import generatePolyBasisLinearModelData

def updateDesignMatrix(old_design_matrix, new_x, n, total_num):
    new_data = np.zeros(n, dtype=float)
    for i in range(n):
        new_data[i] = new_x ** i
    design_matrix = np.append(old_design_matrix, new_data).reshape(total_num, n)
    return design_matrix

def calculatePosteriorMeanAndVar(a, b, y, design_matrix, prior_mean, prior_S):
    # posterior covariance matrix posterior_S = aX^TX + prior_S
    # posterior mean posterior_mean = new_S^-1(aX^Ty + prior_S * prior_mean)
    posterior_S = a * np.matmul(np.transpose(design_matrix), design_matrix) + prior_S
    posterior_mean = np.matmul(np.linalg.inv(posterior_S), (a * np.transpose(design_matrix) * y) + prior_S * prior_mean)

    return posterior_mean, posterior_S

def printResult(x, y, posterior_mean, posterior_S, predict_mean, predict_S):
    print(f"Add data point ({x}, {y}):")
    print(f"\nPosterior mean:")
    for i in range(np.size(posterior_mean)):
        print(f"  {posterior_mean[i]}")
    print(f"\nPosterior variance:")
    size = np.size(posterior_S, 0)
    for i in range(size):
        for j in range(size - 1):
            print(f"  {posterior_S[i][j]},", end = "")
        print(f"  {posterior_S[i][size - 1]}")
    print(f"\nPredictive distribution ~ N({float(predict_mean)}, {float(predict_S)})")

def bayesianLinearRegression():
    w = np.array([1 ,2 ,3 ,4])
    a = 1
    b = 1
    n = 4
    x, y = generatePolyBasisLinearModelData(n, a, w)
    total_num = 1
    design_matrix = np.zeros((1, n), dtype=float)
    for i in range(n):
        design_matrix[0][i] = x ** i

    # first loop
    precision = a * np.matmul(np.transpose(design_matrix), design_matrix) + b * np.identity(n)
    posterior_mean = a * np.matmul(np.linalg.inv(precision), np.transpose(design_matrix)) * y
    posterior_S = np.linalg.inv(precision)
    predict_mean = np.matmul(design_matrix, posterior_mean)
    predict_S = a ** -1 + np.matmul(np.matmul(design_matrix, posterior_S), np.transpose(design_matrix))
    printResult(x, y, posterior_mean, posterior_S, predict_mean, predict_S)

    exit()
