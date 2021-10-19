import numpy as np

"""
An easy to program approximate approach, that relies on the central limit theorem, is as follows:
1. generate 12 uniform U(0,1) deviates,
2. add them all up,
3. subtract 6
the resulting random variable will have approximately standard normal distribution.

We generate a data point which is N~(0,1), but we want to generate a data point which is N~(mean, variance).
We need do linear transformation.
i.e. N~(0,1) -> N~(mean, variance)
Thus, X = mean + sqrt(variance) * generateData.
"""
def generateUnivariateGaussianData(mean, variance):
    # generate data point with N~(0,1)
    s = np.random.uniform(0, 1, 12)
    sum = np.sum(s)
    data = sum - 6

    # transform data point from N~(0,1) to N~(mean, variance)
    Gaussian_data = mean + (variance ** 0.5) * data
    return Gaussian_data


def generatePolyBasisLinearModelData(n, a, w):
    x = np.random.uniform(-1, 1)
    y = 0.0
    for i in range(n):
        y += w[i] * (x ** i)
    y += generateUnivariateGaussianData(0, a)
    return x, y

def dataGenerator():
    generateUnivariateGaussianData(0, 1)
