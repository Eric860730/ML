import ml
import numpy as np

def Gradient_descent(D1, D2):
    pass

def Logistic_regression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
    # Generate D1 and D2
    D1 = np.zeros((N,2))
    D2 = np.zeros((N,2))
    for i in range(N):
        D1[i][0] = ml.generateUnivariateGaussianData(mx1, vx1);
        D1[i][1] = ml.generateUnivariateGaussianData(my1, vy1);
        D2[i][0] = ml.generateUnivariateGaussianData(mx2, vx2);
        D2[i][1] = ml.generateUnivariateGaussianData(my2, vy2);

    print(f"{D1}\n{D2}")
    print(f"{N} {mx1}, {vx1}, {my1}, {vy1}, {mx2}, {vx2}, {my2}, {vy2}")
    d = ml.generateUnivariateGaussianData(1,1);
    print(f"{d}")
    print("in")
    pass
