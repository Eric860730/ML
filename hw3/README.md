# Machine Learning HW2

## Overview
1. Random Data Generator

    a. Univariate gaussian data generator
    
      Given mean 'm' and variance 's', then generate a data point from N(m,s)
      
    b. Polynomial basis linear model generator
    
      Given basis number 'n', variance of error 'a' and Polynomial coefficient 'w', then random product a 'x' between -1.0 and 1.0, which is uniformly distributed, and calculate responsed y.

2. Sequential Estimator 

    Use the input generated by (1,a) to do sequential estimate the mean and variance. Set Bias = 0.05.
3. Bayesian Linear regression

    Use the input generated by (1,b) to do Bayesian linear regression and draw the result.

## Usage

Install the dependencies with Poetry.

``` bash
poetry install --no-root
```

Enter the Poetry shell

``` bash
poetry shell
```

Use the command line arguments to feed the test.csv file, set the degree of the regression model to n, and set the lambda.

``` bash
Usage: python3 main.py hw3-1-a <mean> <variance>
Usage: python3 main.py hw3-1-b <n> <a> <w>
Usage: python3 main.py hw3-2 <mean> <variance>
Usage: python3 main.py hw3-3 <bias int a> <precision int b> <poly_basis int n> <init_prior array w>
```