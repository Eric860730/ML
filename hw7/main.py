import sys
import os
from typing import Tuple
import numpy as np
from PIL import Image

import ml
import config


def usage():
    print("Usage: python3 main.py hw7-1 <method> <mode> <k_neighbors> <kernel_func> <gamma>")
    print("  <method>: 0 for PCA, 1 for LDA")
    print("  <mode>: 0 for naive, 1 for kernel")
    print("  <k_neighbors>: number k of k-NN")
    print("  <kernel_func>: 0 for linear, 1 for RBF")
    print("  <gamma>: if RBF, set gamma for RBF")
    print("Usage: python3 main.py hw7-2")
    exit()


def read_dataset(mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the data according to the corresponding mode, compress the data into np.ndarray and return.
    """
    path = f'./Yale_Face_Database/{mode}'
    files = os.listdir(path)
    images = np.zeros((len(files), config.WIDTH * config.HEIGHT))
    labels = np.zeros(len(files), dtype=int)
    for idx, file in enumerate(files):
        image = Image.open(f'{path}/{file}')
        images[idx] = np.asarray(image.resize(
            (config.WIDTH, config.HEIGHT))).flatten()
        # record the person's index
        labels[idx] = int(file[7:9])

    return images, labels


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        usage()

    if (sys.argv[1] == 'hw7-1'):
        try:
            _ = sys.argv[5]
        except BaseException:
            usage()

        try:
            if(int(sys.argv[5]) == 0):
                if (len(sys.argv) != 6):
                    print(
                        "=== Invalid input: Linear kernel should not provide gamma. ===")
                    usage()
                gamma = 0.0
            elif(int(sys.argv[5]) == 1):
                if (len(sys.argv) != 7):
                    print("=== Invalid input: RBF kernel should provide gamma. ===")
                    usage()
                gamma = float(sys.argv[6])
            else:
                usage()
        except BaseException:
            pass

        train_images, train_labels = read_dataset(mode='Training')
        test_images, test_labels = read_dataset(mode='Testing')
        ml.kernel_eigenfaces(
            train_images, train_labels, test_images, test_labels, int(
                sys.argv[2]), int(
                sys.argv[3]), int(
                sys.argv[4]), int(
                    sys.argv[5]), gamma)

    elif (sys.argv[1] == 'hw7-2'):
        if (len(sys.argv) != 2):
            usage()
        ml.TSNE()

    else:
        usage()
