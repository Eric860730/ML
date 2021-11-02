import numpy as np
import matplotlib.pyplot as plt
from ml import generatePolyBasisLinearModelData

def updateDesignMatrix(new_x, n):
    design_matrix = np.zeros((1, n), dtype=float)
    for i in range(n):
        design_matrix[0][i] = new_x ** i
    return design_matrix

"""
posterior covariance matrix posterior_cov = aX^TX + prior_cov
posterior_mean = posterior_cov^-1(aX^Ty + prior_cov * prior_mean)
"""
def calculatePosteriorMeanAndCov(a, y, design_matrix, prior_mean, prior_cov):
    posterior_cov = a * np.matmul(np.transpose(design_matrix), design_matrix) + prior_cov
    posterior_mean = np.matmul(np.linalg.inv(posterior_cov), (a * np.transpose(design_matrix) * y + np.matmul(prior_cov, prior_mean)))

    return posterior_mean, posterior_cov



"""
predict_mean = Xu
predict_var = 1/a + X*(post_cov^-1)*X^T
since post_cov^-1 = post_var
thus, predict_var = 1/a + X*post_var*X^T
"""
def predictiveDistribution(a, design_matrix, prior_mean, prior_cov):
    predict_mean = np.matmul(design_matrix, prior_mean)
    predict_var = a + np.matmul(np.matmul(design_matrix, np.linalg.inv(prior_cov)), np.transpose(design_matrix))
    return predict_mean, predict_var

"""
when abs(prior_mean - posterior_mean) < 0.00001 and iteration_num > 50
return 0
"""
def checkBias(n, prior_mean, posterior_mean, iteration_num):
    cnt = 0
    if iteration_num < 50:
        return 1
    for i in range(n):
        if abs(prior_mean[i] - posterior_mean[i]) > 0.00001:
            cnt += 1

    return cnt


def printResult(x, y, posterior_mean, posterior_var, predict_mean, predict_var):
    size = np.size(posterior_var, 0)
    print(f"Add data point ({x:.5f}, {y:.5f}):")
    print(f"\nPosterior mean:")
    for i in range(size):
        print(f"  {posterior_mean[i][0]:.10f}")
    print(f"\nPosterior variance:")
    for i in range(size):
        for j in range(size - 1):
            print(f"  {posterior_var[i][j]:.10f},", end = "")
        print(f"  {posterior_var[i][size - 1]:.10f}")
    print(f"\nPredictive distribution ~ N({float(predict_mean):.5f}, {float(predict_var):.5f})\n")


def visualizeRegression(data_x, data_y, data_mean, data_var, a, n, w):
    slice_num = 100
    slice_x = np.linspace(-2.0, 2.0, slice_num)
    matrix_x = np.zeros(shape = (slice_num ,n), dtype = float)
    for i in range(slice_num):
        matrix_x[i] = updateDesignMatrix(slice_x[i], n)

    plt.subplot(221)
    plt.title("Ground truth")
    function = np.poly1d(np.flip(w))
    y = function(slice_x)
    var = a
    drawRegression(slice_x, y, var)

    plt.subplot(222)
    plt.title("Predict result")
    function = np.poly1d(np.flip(np.reshape(data_mean[2], n)))
    y = function(slice_x)
    var = np.zeros(slice_num)
    for i in range(slice_num):
        var[i] = a + np.matmul(np.matmul(matrix_x[i], data_var[2]), np.transpose(matrix_x[i]))
    plt.scatter(data_x, data_y, s=6.0)
    drawRegression(slice_x, y, var)

    plt.subplot(223)
    plt.title("After 10 incomes")
    function = np.poly1d(np.flip(np.reshape(data_mean[0], n)))
    y = function(slice_x)
    var = np.zeros(slice_num)
    for i in range(slice_num):
        var[i] = a + np.matmul(np.matmul(matrix_x[i], data_var[0]), np.transpose(matrix_x[i]))
    plt.scatter(data_x[0:10], data_y[0:10], s=6.0)
    drawRegression(slice_x, y, var)

    plt.subplot(224)
    plt.title("After 50 incomes")
    function = np.poly1d(np.flip(np.reshape(data_mean[1], n)))
    y = function(slice_x)
    var = np.zeros(slice_num)
    for i in range(slice_num):
        var[i] = a + np.matmul(np.matmul(matrix_x[i], data_var[1]), np.transpose(matrix_x[i]))
    plt.scatter(data_x[0:50], data_y[0:50], s=6.0)
    drawRegression(slice_x, y, var)

    plt.tight_layout()
    plt.show()


def drawRegression(x, y, var):
    plt.plot(x, y, color = 'black')
    plt.plot(x, y + var, color = 'red')
    plt.plot(x, y - var, color = 'red')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 30.0)


def bayesianLinearRegression(a, b, n, w):
    # init data
    data_x = np.array([])
    data_y = np.array([])
    data_mean = np.array([])
    data_var = np.array([])
    prior_mean = np.zeros(n)
    prior_cov = np.identity(n)

    # add point
    x, y = generatePolyBasisLinearModelData(n, a, w)
    data_x = np.append(data_x, x)
    data_y = np.append(data_y, y)
    design_matrix = updateDesignMatrix(x, n)

    # first loop
    iteration_num = 1
    posterior_cov = a * np.matmul(np.transpose(design_matrix), design_matrix) + b * np.identity(n)
    posterior_mean = a * np.matmul(np.linalg.inv(posterior_cov), np.transpose(design_matrix)) * y
    posterior_var = np.linalg.inv(posterior_cov)
    predict_mean = np.matmul(design_matrix, prior_mean)
    predict_var = a + np.matmul(np.matmul(design_matrix, posterior_var), np.transpose(design_matrix))
    printResult(x, y, posterior_mean, posterior_var, predict_mean, predict_var)

    while checkBias(n, prior_mean, posterior_mean, iteration_num):
        prior_cov = posterior_cov
        prior_mean = posterior_mean
        iteration_num += 1
        x, y = generatePolyBasisLinearModelData(n, a, w)
        data_x = np.append(data_x, x)
        data_y = np.append(data_y, y)
        design_matrix = updateDesignMatrix(x, n)
        posterior_mean, posterior_cov = calculatePosteriorMeanAndCov(a, y, design_matrix, prior_mean, prior_cov)
        posterior_var = np.linalg.inv(posterior_cov)
        predict_mean, predict_var = predictiveDistribution(a, design_matrix, prior_mean, prior_cov)
        printResult(x, y, posterior_mean, posterior_var, predict_mean, predict_var)

        if iteration_num == 10 or iteration_num == 50:
            data_mean = np.append(data_mean, posterior_mean)
            data_var = np.append(data_var, posterior_var)

    data_mean = np.append(data_mean, posterior_mean)
    data_var = np.append(data_var, posterior_var)
    data_mean = np.reshape(data_mean, (3, n))
    data_var = np.reshape(data_var, (3, n, n))

    visualizeRegression(data_x, data_y, data_mean, data_var, a, n, w)
    exit()

