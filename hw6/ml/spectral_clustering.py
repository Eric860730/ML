import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from ml.kernel_kmeans import computeKernel
from ml.kernel_kmeans import setColor
from ml.kernel_kmeans import getCurrentImage
from scipy.spatial.distance import cdist


# python rich for debug
from rich.traceback import install
install(show_locals=True)


def computeLaplacian(matrix_W: np.ndarray):
    matrix_D = np.zeros((matrix_W.shape))
    for row in range(matrix_W.shape[0]):
        matrix_D[row, row] += np.sum(matrix_W[row])
    matrix_L = matrix_D - matrix_W

    return matrix_D, matrix_L


def computeMatrixU(matrix_W: np.ndarray, num_cluster: int, cut: int):
    # compute Laplacian for get matrix L and degree matrix D
    matrix_D, matrix_L = computeLaplacian(matrix_W)

    # Normalized cut
    if cut == 0:
        # Normalized Laplacian matrix
        for i in range(len(matrix_D)):
            matrix_D[i, i] = 1.0 / np.sqrt(matrix_D[i, i])
        matrix_L = np.matmul(np.matmul(matrix_D, matrix_L), matrix_D)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_L)
    eigenvectors = eigenvectors.T

    # Sort eigenvalues for find the index of nonzero eigenvalues
    sort_idx = np.argsort(eigenvalues)
    mask = eigenvalues[sort_idx] > 0
    sort_idx = sort_idx[mask]
    matrix_U = eigenvectors[sort_idx[:num_cluster]].T

    return matrix_U


def initCenter(
        matrix_U: np.ndarray,
        num_row: int,
        num_col: int,
        num_cluster: int,
        method: int):
    # Random method
    if method == 0:
        return matrix_U[np.random.choice(num_row * num_col, num_cluster)]

    # kmeans++ method
    else:
        # Get grid indices, shape = (2, num_row, num_col)
        grid = np.indices((num_row, num_col))

        # Combine grid indices to np.ndarray
        grid_indices = np.hstack(
            (grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)))

        # Randomly pick first center
        num_pixel = num_row * num_col
        center = [grid_indices[np.random.choice(num_pixel, 1)[0]].tolist()]

        # Pick other center
        for num_center in range(num_cluster - 1):
            dist = np.zeros(num_pixel)
            for i in range(len(grid_indices)):
                min_dist = np.Inf
                for j in range(num_center + 1):
                    cur_dist = np.linalg.norm(grid_indices[i] - center[j])
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                dist[i] = min_dist
            dist /= np.sum(dist)
            center.append(grid_indices[np.random.choice(
                num_pixel, 1, p=dist)[0]].tolist())
        feature_center = []
        for i in range(num_cluster):
            feature_center.append(
                matrix_U[center[i][0] * num_row + center[i][1], :])
        feature_center = np.array(feature_center)

        return feature_center


def computeClustering(
        matrix_U: np.ndarray,
        current_center: np.ndarray,
        num_pixel: int,
        num_cluster: int):
    cluster = np.zeros(num_pixel, dtype=int)
    for i in range(num_pixel):
        dist = np.zeros(num_cluster)
        for j in range(len(current_center)):
            dist[j] = np.linalg.norm((matrix_U[i] - current_center[j]), ord=2)
        cluster[i] = np.argmin(dist)

    return cluster


def computeCenter(
        matrix_U: np.ndarray,
        new_cluster: np.ndarray,
        num_cluster: int):
    new_centers = []
    for c in range(num_cluster):
        points_in_center = matrix_U[new_cluster == c]
        new_center = np.average(points_in_center, axis=0)
        new_centers.append(new_center)

    return np.array(new_centers)


def computeKmeans(
        center: np.ndarray,
        matrix_U: np.ndarray,
        num_row: int,
        num_col: int,
        num_cluster: int,
        method: int,
        cut: int,
        image_num: int):
    # Set color
    color = setColor(num_cluster)

    # Save all image to image array
    image = []

    # compute k-means
    current_center = center.copy()
    num_pixel = num_row * num_col
    new_cluster = np.zeros(num_pixel, dtype = int)
    iteration = 100
    for _ in range(iteration):
        # Compute new cluster
        new_cluster = computeClustering(
            matrix_U, current_center, num_pixel, num_cluster)

        # Compute new center
        new_center = computeCenter(matrix_U, new_cluster, num_cluster)

        # Save current status to image array
        image.append(getCurrentImage(num_row, num_col, new_cluster, color))

        if np.linalg.norm((new_center - current_center), ord=2) < 1e-2:
            break

        current_center = new_center.copy()

    # Save begin and final png
    filename_png_start = getFilename(
        image_num, num_cluster, method, cut, '_Start', 'png')

    # Save gif
    filename_gif = getFilename(image_num, num_cluster, method, cut, '', 'gif')
    if len(image) > 1:
        image[0].save(filename_png_start)
        filename_png_end = getFilename(
            image_num, num_cluster, method, cut, '_End', 'png')
        image[-1].save(filename_png_end)
        image[0].save(filename_gif,
                      save_all=True,
                      append_images=image[1:],
                      optimize=False,
                      loop=0,
                      duration=100)
    else:
        image[0].save(filename_gif)
        image[0].save(filename_png_start)

    return new_cluster


def getFilename(
        image_num: int,
        num_cluster: int,
        method: int,
        cut: int,
        time: str,
        ftype: str):
    if ftype == 'gif':
        dirname = './output_images/spectral_clustering_gif'
    else:
        dirname = './output_images/spectral_clustering_png'
    if method == 0:
        m = 'Random'
    elif method == 1:
        m = 'Kmeans++'
    if cut == 0:
        c = 'Normalized'
    elif cut == 1:
        c = 'Ratio'
    filename = f'{dirname}/image_{image_num}_cluster_{num_cluster}_{m}_{c}{time}.{ftype}'
    os.makedirs(dirname, exist_ok=True)

    return filename


def computeSpectralClustering(
        matrix_U: np.ndarray,
        num_row: int,
        num_col: int,
        num_cluster: int,
        method: int,
        cut: int,
        image_num: int):
    # init center
    print("=== Spectral Clustering - Init Center ===")
    center = initCenter(matrix_U, num_row, num_col, num_cluster, method)

    # K-means
    print("=== Spectral Clustering - K-means ===")
    cluster = computeKmeans(
        center,
        matrix_U,
        num_row,
        num_col,
        num_cluster,
        method,
        cut,
        image_num)

    # Plot the result in the eigenspace
    if num_cluster < 4:
        plotEigenspace(matrix_U, cluster, method, cut, image_num, num_cluster)


def plotEigenspace(
        matrix_U: np.ndarray,
        cluster: np.ndarray,
        method: int,
        cut: int,
        image_num: int,
        num_cluster: int):
    if num_cluster == 2:
        color = ['red', 'blue']
    else:
        color = ['red', 'blue', 'green']

    if method == 0:
        m = 'Random'
    else:
        m = 'Kmeans++'

    if cut == 0:
        c = 'Normalized'
    else:
        c = 'Ratio'

    plt.title(f'image{image_num}_cluster{num_cluster}_{m}_{c}')
    for i in range(len(matrix_U)):
        plt.scatter(matrix_U[i][0], matrix_U[i][1], c=color[cluster[i]])

    # Save the figure
    dirname = './output_images/spectral_clustering/eigenspace'
    filename = f'{dirname}/eigenspace_image{image_num}_cluster{num_cluster}_{m}_{c}.png'
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(filename)


def spectral_clustering(
        image: np.ndarray,
        image_num: int,
        gamma_s: float,
        gamma_c: float,
        num_cluster: int,
        method: int,
        cut: int):
    num_row, num_col, num_color = image.shape

    # compute kernel
    print("=== Compute Kernel ===")
    kernel = computeKernel(
        image,
        gamma_s,
        gamma_c,
        num_row,
        num_col,
        num_color)

    # compute matrix U
    print("=== Compute Matrix U ===")
    matrix_U = computeMatrixU(kernel, num_cluster, cut)

    # Normalized cut
    if cut == 0:
        row_sum = np.sum(matrix_U, axis=1)
        for row in range(len(matrix_U)):
            matrix_U[row, :] /= row_sum[row]

    print("=== Compute Spectral Clustering ===")
    computeSpectralClustering(
        matrix_U,
        num_row,
        num_col,
        num_cluster,
        method,
        cut,
        image_num)
