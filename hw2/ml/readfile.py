import sys
import os
import struct
import numpy as np


def read_data(root):
    train_label_path = os.path.join(root, 'train-labels-idx1-ubyte')
    train_image_path = os.path.join(root, 'train-images-idx3-ubyte')
    test_label_path = os.path.join(root, 't10k-labels-idx1-ubyte')
    test_image_path = os.path.join(root, 't10k-images-idx3-ubyte')


    with open(train_label_path, 'rb') as file:
        magic, item_num = struct.unpack('>II', file.read(8))
        if (magic != 2049):
            print('Oops! The magic number is not equal to 2049!')
            exit()
        else:
            train_label = np.fromfile(file, dtype = np.dtype('B'), count = -1)


    with open(train_image_path, 'rb') as file:
        magic, image_num, row, col = struct.unpack('>IIII', file.read(16))
        if (magic != 2051):
            print('Oops! The magic number is not equal to 2051!')
            exit()
        else:
            data = np.fromfile(file, dtype = np.dtype('B'), count = -1)
            train_image = data.reshape((image_num, row, col))


    with open(test_label_path, 'rb') as file:
        magic, item_num = struct.unpack('>II', file.read(8))
        if (magic != 2049):
            print('Oops! The magic number is not equal to 2049!')
            exit()
        else:
            test_label = np.fromfile(file, dtype = np.dtype('B'), count = -1)


    with open(test_image_path, 'rb') as file:
        magic, image_num, row, col = struct.unpack('>IIII', file.read(16))
        if (magic != 2051):
            print('Oops! The magic number is not equal to 2051!')
            exit()
        else:
            data = np.fromfile(file, dtype = np.dtype('B'), count = -1)
            test_image = data.reshape((image_num, row, col))

    return train_label, train_image, test_label, test_image
