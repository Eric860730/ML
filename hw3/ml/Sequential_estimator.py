import numpy as np
from ml import generateUnivariateGaussianData

Bias = 0.05

def checkBias(m, s, new_mean, new_variance):
    if abs(m - new_mean) < Bias and abs(s - new_variance) < Bias:
        return True
    else:
        return False

def sequentialEstimator(m, s):
    print(f"Data point source function: N({m}, {s})")
    print()
    n = 1
    add_point = generateUnivariateGaussianData(m, s)
    old_mean = m
    old_variance = s
    old_mean_sq = s
    n += 1
    new_mean = old_mean + (add_point - old_mean) / n
    new_mean_sq = old_mean_sq + (add_point - old_mean) * (add_point - new_mean)
    new_variance = new_mean_sq / n
    print(f"Add data point: {add_point}")
    print(f"Mean = {new_mean}\tVariance = {new_variance}")

    while not checkBias(m, s, new_mean, new_variance):
        old_mean = new_mean
        old_variance = new_variance
        old_mean_sq = new_mean_sq
        add_point = generateUnivariateGaussianData(m, s)
        n += 1
        new_mean = old_mean + (add_point - old_mean) / n
        new_mean_sq = old_mean_sq + (add_point - old_mean) * (add_point - new_mean)
        new_variance = new_mean_sq / n
        print(f"Add data point: {add_point}")
        print(f"Mean = {new_mean}\tVariance = {new_variance}")
