import numpy as np
from ml import generatePolyBasisLinearModelData

def updateDesignMatrix(old_design_matrix, new_x, n, total_num):
    new_data = np.zeros(n, dtype=float)
    for i in range(n):
        new_data[i] = new_x ** i
    design_matrix = np.append(old_design_matrix, new_data).reshape(total_num, n)
    return design_matrix

def calculatePosteriorMeanAndCov(a, b, y, design_matrix, prior_mean, prior_cov):
    # posterior covariance matrix posterior_var = aX^TX + prior_var
    # posterior mean posterior_mean = new_var^-1(aX^Ty + prior_var * prior_mean)
    posterior_cov = a * np.matmul(np.transpose(design_matrix), design_matrix) + prior_cov
    print(design_matrix)
    print(prior_mean)
    print(prior_cov)
    posterior_mean = np.matmul(np.linalg.inv(posterior_cov), (a * np.transpose(design_matrix) * y + np.matmul(prior_cov, prior_mean)))

    return posterior_mean, posterior_cov


def printResult(x, y, posterior_mean, posterior_var, predict_mean, predict_var):
    size = np.size(posterior_var, 0)
    print(f"Add data point ({x}, {y}):")
    print(f"\nPosterior mean:")
    for i in range(size):
        print(f"  {posterior_mean[i]}")
    print(f"\nPosterior variance:")
    for i in range(size):
        for j in range(size - 1):
            print(f"  {posterior_var[i][j]},", end = "")
        print(f"  {posterior_var[i][size - 1]}")
    print(predict_mean)
    print(predict_var)
    print(f"\nPredictive distribution ~ N({float(predict_mean)}, {float(predict_var)})\n")

def predictiveDistribution(a, design_matrix, prior_mean, prior_cov):
    predict_mean = np.matmul(design_matrix, prior_mean)[0][0]
    predict_var = (a ** -1 + np.matmul(np.matmul(design_matrix, np.linalg.inv(prior_cov)), np.transpose(design_matrix)))[0][0]
    return predict_mean, predict_var

def bayesianLinearRegression():
    w = np.array([1 ,2 ,3 ,4])
    a = 1
    b = 1
    n = 4
    data_x = np.array([])
    data_y = np.array([])
    prior_mean = np.zeros(n)
    prior_cov = np.identity(n)
    x, y = generatePolyBasisLinearModelData(n, a, w)
    data_x = np.append(data_x, x)
    data_y = np.append(data_y, y)
    total_num = 1
    design_matrix = np.zeros((1, n), dtype=float)
    for i in range(n):
        design_matrix[0][i] = x ** i

    # first loop
    posterior_cov = a * np.matmul(np.transpose(design_matrix), design_matrix) + b * np.identity(n)
    posterior_mean = a * np.matmul(np.linalg.inv(posterior_cov), np.transpose(design_matrix)) * y
    posterior_var = np.linalg.inv(posterior_cov)
    predict_mean = np.matmul(design_matrix, prior_mean)
    predict_var = a ** -1 + np.matmul(np.matmul(design_matrix, np.linalg.inv(prior_cov)), np.transpose(design_matrix))
    printResult(x, y, posterior_mean, posterior_var, predict_mean, predict_var)

    while checkBias(n, prior_mean, posterior_mean, total_num):
        prior_cov = posterior_cov
        prior_mean = posterior_mean
        prior_var = posterior_var
        x, y = generatePolyBasisLinearModelData(n, a, w)
        data_x = np.append(data_x, x)
        data_y = np.append(data_y, y)
        total_num += 1
        design_matrix = updateDesignMatrix(design_matrix, x, n, total_num)
        posterior_mean, posterior_cov = calculatePosteriorMeanAndCov(a, b, y, design_matrix, prior_mean, prior_cov)
        posterior_var = np.linalg.inv(posterior_cov)
        predict_mean, predict_var = predictiveDistribution(a, design_matrix, prior_mean, prior_var)
        printResult(x, y, posterior_mean, posterior_var, predict_mean, predict_var)

    print("converge")



def checkBias(n, prior_mean, posterior_mean, total_num):
    cnt = 0
    if total_num < 50:
        return 1
    for i in range(n):
        if abs(prior_mean[i] - posterior_mean[i]) > 0.00001:
            cnt += 1

    return cnt




