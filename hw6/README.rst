=============
Machine Learning HW6
=============

Overview
---------

1. Kernel kmeans
    - Automatically read the image in ./data/
    - Doing kernel kmeans to classified the points in the image.
    - Saved the results in ./output_images/

2. Spectral clustering
    - Automatically read the image in ./data/
    - Doing Spectral clustering to classified the points in the image.
    - Saved the results in ./output_images/

Environment
---------

Requirment
^^^^^^^^^

- python3 >= 3.8
- numpy >= 1.21
- scipy >= 1.6.1
- matplotlib >= 3.4
- pyqt5 >= 5.15
- libsvm-official >= 3.25

Build virtual environment
^^^^^^^^^

Build the virtual environment with Poetry::

    poetry install --no-dev


Enter the Poetry shell::

    poetry shell


Run
---------

Run hw6 with the following command::

    Usage: python3 main.py hw6-1 <image number> <gamma_s> <gamma_c> <k(number of cluster)> <method(0: random, 1: k-means++)>
    Usage: python3 main.py hw6-2 <image number> <gamma_s> <gamma_c> <k(number of cluster)> <method(0: random, 1: k-means++)> <cut(0: normalized, 1: ratio)>
