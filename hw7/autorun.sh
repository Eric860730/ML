#!/bin/bash
# Usage: python3 main.py hw7-1 <method> <mode> <k_neighbors> <kernel_func> <gamma>
# Naive PCA k = 5
python3 main.py hw7-1 0 0 5 0
python3 main.py hw7-1 0 1 5 0
python3 main.py hw7-1 0 1 5 1 0.000001
python3 main.py hw7-1 1 0 5 0
python3 main.py hw7-1 1 1 5 0
python3 main.py hw7-1 1 1 5 1 0.000001
