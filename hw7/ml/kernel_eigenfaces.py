import config
import os
import numpy as np
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# python rich for debug
from rich.traceback import install
install(show_locals=True)


def computeSimplePCACov(train_images: np.ndarray,
                       mean_images: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix of Simple PCA
    """
    difference_images = train_images - mean_images
    cov_matrix = difference_images.T.dot(difference_images)

    return cov_matrix


def computeKernelPCACov(train_images: np.ndarray,
                        kernel_type: int, gamma: float) -> np.ndarray:
    """
    Compute the covariance matrix of kernel PCA
    """
    if kernel_type == 0:
        # Linear
        kernel = train_images.T.dot(train_images)
    else:
        # RBF
        kernel = np.exp(-gamma * cdist(train_images.T,
                        train_images.T, 'sqeuclidean'))

    matrix_n = np.ones((config.WIDTH * config.HEIGHT,
                        config.WIDTH * config.HEIGHT),
                       dtype=float) / (config.WIDTH * config.HEIGHT)
    cov_matrix = kernel - \
        matrix_n.dot(kernel) - kernel.dot(matrix_n) + \
        matrix_n.dot(kernel).dot(matrix_n)

    return cov_matrix


def findPrincipalEigenvectors(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Find 25(in this work) first largest eigenvectors as principal components.
    """
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Find 25 first largest eigenvectors
    sort_index = np.argsort(-eigenvalues)
    principal_eigenvectors = eigenvectors[:, sort_index[0:25]].real

    return principal_eigenvectors


def saveFigure(method: int, mode: int, kernel_type: int, title: str) -> None:
    """
    Save current figure in ./output_images/
    method: 0 for PCA, 1 for LDA
    """
    dirname = f'./output_images/{"LDA" if method else "PCA"}'
    os.makedirs(dirname, exist_ok=True)
    if mode == 0:
        # Simple
        title = f'Simple_{title}_{config.WIDTH}x{config.HEIGHT}'
    else:
        # Kernel
        if kernel_type == 0:
            # Linear
            title = f'Linear_Kernel_{title}_{config.WIDTH}x{config.HEIGHT}'
        else:
            # RBF
            title = f'RBF_Kernel_{title}_{config.WIDTH}x{config.HEIGHT}'
    filename = f'{dirname}/{title}.png'
    plt.savefig(filename)


def drawEigenfaces(eigenfaces: np.ndarray, title: str) -> None:
    """
    Show the results of eigenfaces.
    """
    plt.figure(title)
    # Five eigenfaces per row
    num_row = int(eigenfaces.shape[0] / 5)
    for idx in range(len(eigenfaces)):
        plt.subplot(num_row, int((len(eigenfaces) + 1) / num_row), idx + 1)
        plt.axis('off')
        plt.imshow(eigenfaces[idx], cmap='gray')


def reconstructEigenfaces(train_images: np.ndarray,
                          principal_eigenvectors: np.ndarray) -> np.ndarray:
    """
    Randomly choose 10 train_images to reconstruct eigenfaces.
    """
    # Randomly choose 10 train_images
    chosen_index = random.sample(range(len(train_images)), 10)

    reconstruction_eigenfaces = train_images[chosen_index].dot(
        principal_eigenvectors).dot(principal_eigenvectors.T)

    return reconstruction_eigenfaces


def decorrelateImages(num_of_images: int, images: np.ndarray,
                      principal_eigenvectors: np.ndarray) -> np.ndarray:
    """
    Decorrelate unnecessary components.
    """
    decorrelate_images = np.zeros((num_of_images, 25))
    for idx, image in enumerate(images):
        decorrelate_images[idx, :] = image.dot(principal_eigenvectors)

    return decorrelate_images


def classifyAndRecognize(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray,
                         test_labels: np.ndarray, principal_eigenvectors: np.ndarray, num_k: int) -> None:
    # Get the number of train_images and test_images
    num_of_train = len(train_images)
    num_of_test = len(test_images)

    decorrelated_train = decorrelateImages(
        num_of_train, train_images, principal_eigenvectors)
    decorrelated_test = decorrelateImages(
        num_of_test, test_images, principal_eigenvectors)
    error = 0
    distance = np.zeros(num_of_train)
    for test_idx in range(num_of_test):
        for train_idx in range(num_of_train):
            distance[train_idx] = np.linalg.norm(
                decorrelated_test[test_idx] - decorrelated_train[train_idx])
        min_distance = np.argsort(distance)[:num_k]
        predict = np.argmax(np.bincount(train_labels[min_distance]))
        if predict != test_labels[test_idx]:
            error += 1
    print(
        f'Error count: {error}, Accuracy: {(len(test_labels) - error) / num_of_test}')


def computePCA(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray,
               test_labels: np.ndarray, mode: int, num_k: int, kernel_type: int, gamma: float) -> None:
    """
    compute Principal Components Analysis
    """
    # Compute mean of train_images
    mean_images = np.sum(train_images, axis=0) / len(train_images)

    if mode == 0:
        # Simple PCA
        cov_matrix = computeSimplePCACov(train_images, mean_images)

        # Recorded the input config
        print(f'Simple PCA with {num_k}-NN')
    else:
        # Kernel PCA
        cov_matrix = computeKernelPCACov(train_images, kernel_type, gamma)

        # Recorded the input config
        print(
            f'{"RBF" if kernel_type else "Linear"} Kernel PCA with {num_k}-NN',
            end='')
        if kernel_type:
            print(f' gamma {gamma}')
        else:
            print('')

    # Find 25 first largest principal eigenvectors
    principal_eigenvectors = findPrincipalEigenvectors(cov_matrix)

    # Get eigenfaces and show it
    eigenfaces = principal_eigenvectors.T
    drawEigenfaces(
        eigenfaces.reshape(
            25,
            config.HEIGHT,
            config.WIDTH),
        'Original PCA Eigenfaces')
    saveFigure(0, mode, kernel_type, 'PCA_Original_Eigenfaces')

    # Randomly choose 10 train eigenfaces to reconstruction and show it
    reconstruction_eigenfaces = reconstructEigenfaces(
        train_images, principal_eigenvectors)
    drawEigenfaces(
        reconstruction_eigenfaces.reshape(
            10,
            config.HEIGHT,
            config.WIDTH),
        'Reconstruct PCA Eigenfaces')
    saveFigure(0, mode, kernel_type, 'PCA_Reconstruct_Eigenfaces')

    # Classify test_images for face recognize.
    classifyAndRecognize(
        train_images,
        train_labels,
        test_images,
        test_labels,
        principal_eigenvectors,
        num_k)

    # Plot the results
    # plt.tight_layout()
    # plt.show()


def computeSimpleLDACov(count_num_of_class: int, train_images: np.ndarray,
                       train_labels: np.ndarray) -> np.ndarray:
    # Compute global mean
    global_mean = np.mean(train_images, axis=0)
    image_size = config.HEIGHT * config.WIDTH

    # Compute mean of each class
    class_num = len(count_num_of_class)
    class_mean = np.zeros((class_num, image_size))
    for label_idx in range(class_num):
        class_mean[label_idx, :] = np.mean(
            train_images[train_labels == label_idx + 1], axis=0)

    # Compute between-class scatter B
    scatter_b = np.zeros((image_size, image_size), dtype=float)
    for idx in range(len(count_num_of_class)):
        # difference = mj - m
        difference = (class_mean[idx] - global_mean).reshape((image_size, 1))
        scatter_b += count_num_of_class[idx] * difference.dot(difference.T)

    # Compute within-class scatter
    scatter_w = np.zeros((image_size, image_size), dtype=float)
    for idx in range(len(class_mean)):
        # difference = xi - mj, where i is belongs to Cj
        difference = train_images[train_labels == idx + 1] - class_mean[idx]
        scatter_w += difference.T.dot(difference)

    # Compute lambda = Sw^-1 * Sb
    cov_matrix = np.linalg.pinv(scatter_w).dot(scatter_b)

    return cov_matrix


def computeKernelLDACov(count_num_of_class: int, train_images: np.ndarray,
                        train_labels: np.ndarray, kernel_type: int, gamma: float):
    """
    reference: https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis
    matrix N: sigma Kk(I - 1/num_k * I)Kk^T, Kk means k-th kernel
    """
    class_num = len(count_num_of_class)
    image_num = len(train_images)
    image_size = config.HEIGHT * config.WIDTH

    # Compute kernel
    if kernel_type == 0:
        # Linear
        kernel_per_class = np.zeros((class_num, image_size, image_size))
        for idx in range(class_num):
            tmp_image = train_images[train_labels == idx + 1]
            kernel_per_class[idx] = tmp_image.T.dot(tmp_image)
        kernel_all = train_images.T.dot(train_images)
    else:
        # RBF
        kernel_per_class = np.zeros((class_num, image_size, image_size))
        for idx in range(class_num):
            tmp_image = train_images[train_labels == idx + 1]
            kernel_per_class[idx] = np.exp(-gamma *
                                           cdist(tmp_image.T, tmp_image.T, 'sqeuclidean'))
        kernel_all = np.exp(-gamma * cdist(train_images.T,
                            train_images.T, 'sqeuclidean'))

    # Compute matrix N
    matrix_n = np.zeros((image_size, image_size))
    identity_matrix = np.identity(image_size)
    for idx in range(class_num):
        matrix_n += kernel_per_class[idx].dot(identity_matrix - (
            1 / count_num_of_class[idx]) * identity_matrix).dot(kernel_per_class[idx].T)

    # Compute matrix M
    M_i = np.zeros((class_num, image_size))
    for idx, kernel in enumerate(kernel_per_class):
        for row_idx, row in enumerate(kernel):
            M_i[idx, row_idx] = np.sum(row) / count_num_of_class[idx]
    M_mean = np.zeros(image_size)
    for idx, row in enumerate(kernel_all):
        M_mean[idx] = np.sum(row) / image_num
    matrix_m = np.zeros((image_size, image_size))
    for idx, num in enumerate(count_num_of_class):
        difference = (M_i[idx] - M_mean).reshape((image_size, 1))
        matrix_m += num * difference.dot(difference.T)

    # Compute alpha = N^-1 * M
    cov_matrix = np.linalg.pinv(matrix_n).dot(matrix_m)

    return cov_matrix


def computeLDA(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray,
               test_labels: np.ndarray, mode: int, num_k: int, kernel_type: int, gamma: float) -> None:
    """
    compute Linear Discriminative Analysis
    """
    _, count_num_of_class = np.unique(train_labels, return_counts=True)

    if mode == 0:
        # Simple LDA
        cov_matrix = computeSimpleLDACov(
            count_num_of_class, train_images, train_labels)

        # Recorded the input config
        print(f'Simple LDA with {num_k}-NN')
    else:
        # Kernel LDA
        cov_matrix = computeKernelLDACov(
            count_num_of_class,
            train_images,
            train_labels,
            kernel_type,
            gamma)

        # Recorded the input config
        print(
            f'{"RBF" if kernel_type else "Linear"} Kernel LDA with {num_k}-NN',
            end='')
        if kernel_type:
            print(f' gamma {gamma}')
        else:
            print('')

    # Find 25 first largest principal eigenvectors
    principal_eigenvectors = findPrincipalEigenvectors(cov_matrix)

    # Get eigenfaces and show it
    eigenfaces = principal_eigenvectors.T
    drawEigenfaces(
        eigenfaces.reshape(
            25,
            config.HEIGHT,
            config.WIDTH),
        'Original LDA Eigenfaces')
    saveFigure(1, mode, kernel_type, 'LDA_Original_Eigenfaces')

    # Randomly choose 10 train eigenfaces to reconstruction and show it
    reconstruction_eigenfaces = reconstructEigenfaces(
        train_images, principal_eigenvectors)
    drawEigenfaces(
        reconstruction_eigenfaces.reshape(
            10,
            config.HEIGHT,
            config.WIDTH),
        'Reconstruct LDA Eigenfaces')
    saveFigure(1, mode, kernel_type, 'LDA_Reconstruct_Eigenfaces')

    # Classify test_images for face recognize.
    classifyAndRecognize(
        train_images,
        train_labels,
        test_images,
        test_labels,
        principal_eigenvectors,
        num_k)

    # Plot the results
    # plt.tight_layout()
    # plt.show()


def kernel_eigenfaces(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray,
                      test_labels: np.ndarray, method: int, mode: int, num_k: int, kernel_type: int, gamma: float) -> None:
    if method == 0:
        # PCA
        print(f'\n===== Compute PCA start =====')
        computePCA(
            train_images,
            train_labels,
            test_images,
            test_labels,
            mode,
            num_k,
            kernel_type,
            gamma)
    else:
        # LDA
        print(f'\n===== Compute LDA start =====')
        computeLDA(
            train_images,
            train_labels,
            test_images,
            test_labels,
            mode,
            num_k,
            kernel_type,
            gamma)
