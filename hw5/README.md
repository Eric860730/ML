# Machine Learning HW5

## Overview
1. Gaussian Process Regression
    - Automatically read the data in ./data/input.data
    - Compute the Gaussian Process Regression to predict the distribution of f and visualize the result.
    - Optimize the kernel parameters by minimizing negative marginal log-likelihood, and visualize the result.
2. SVM
    - Automatically read the data in ./data/*.csv
    - Compare the performance between different kernel functions.
    - Use the grid search method to find the best parameters of each kernel functions.
    - Use the grid search method to find the best parameters of user defined kernel function (linear + RBF) and compare its performance with other kernel functions.

## Environment

### Requirment
python3 >= 3.8
numpy >= 1.21
scipy >= 1.6.1
matplotlib >= 3.4
pyqt5 >= 5.15
libsvm-official >= 3.25

### Build virtual environment
Build the virtual environment with Poetry.

``` bash
poetry install --no-root
```

Enter the Poetry shell

``` bash
poetry shell
```

## Run

Run hw5 with the following command.

``` bash
Usage: python3 main.py hw5-1
Usage: python3 main.py hw5-2
```

## Reference
https://stats.stackexchange.com/questions/407254/intuition-behind-the-length-scale-of-the-rational-quadratic-kernel
https://www.itread01.com/content/1549816773.html
