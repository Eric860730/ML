=============
Machine Learning HW7
=============

Overview
---------

1. Kernel eigenfaces

    - Automatically read the data in ./Yale_Face_Database/
    - Using both PCA and LDA method to training eigenfaces and classify the testing faces.
    - Store the results figure in ./output_images/LDA/ and ./output_images/PCA/
    - You can modify config.py to determine the compressed size of the image you want.

2. t-SNE and symmetric-SNE

    - Automatically read the data in ./mnist_data/
    - Run both t-SNE and symmetric-SNE and then store the classified results in ./output_images/SNE/
    - Also store the figures of the similarity of Low-D and High-D in ./output_images/SNE/

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

Build the virtual environment with Poetry.

``` bash
poetry install --no-dev

```

Enter the Poetry shell

``` bash
poetry shell

```

Run
---------

Run hw7 with the following command.

``` bash
Usage: python3 main.py hw7-1 <method> <mode> <k_neighbors> <kernel_func> <gamma>
      <method>: 0 for PCA, 1 for LDA
      <mode>: 0 for naive, 1 for kernel
      <k_neighbors>: number k of k-NN
      <kernel_func>: 0 for linear, 1 for RBF
      <gamma>: if RBF, set gamma for RBF
Usage: python3 main.py hw7-2

```

Source
---------
| [1]: `tsne.py and mnist data <https://lvdmaaten.github.io/tsne/>`
