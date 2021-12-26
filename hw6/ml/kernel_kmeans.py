import numpy as np
import os
from scipy.spatial.distance import cdist
from PIL import Image

# python rich for debug
from rich.traceback import install
install(show_locals=True)


def computeKernel(
        image: np.ndarray,
        gamma_s: float,
        gamma_c: float,
        num_row: int,
        num_col: int,
        num_color: int):
    # Transform image shape
    image_data = image.reshape(num_row * num_col, num_color)

    # Compute color distance
    color_dist = cdist(image_data, image_data, 'sqeuclidean')

    # Get grid indices, shape = (2, num_row, num_col)
    grid = np.indices((num_row, num_col))

    # Combine grid indices to np.ndarray
    grid_indices = np.hstack((grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)))

    # Compute spatial distance
    spatial_dist = cdist(grid_indices, grid_indices, 'sqeuclidean')

    return np.multiply(np.exp(-gamma_s * spatial_dist),
                       np.exp(-gamma_c * color_dist))


def chooseCenter(num_row: int, num_col: int, num_cluster: int, method: int):
    if method == 0:
        # Random method
        return np.random.choice(num_row, (num_cluster, 2))
    elif method == 1:
        # kmeans++ method
        # Get grid indices, shape = (2, num_row, num_col)
        grid = np.indices((num_row, num_col))

        # Combine grid indices to np.ndarray
        grid_indices = np.hstack(
            (grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)))

        # Randomly pick first center
        num_pixel = num_row * num_col
        random = np.random.choice(num_pixel, 1)
        center = []
        center.append(grid_indices[random[0]])

        # Pick other center
        for num_center in range(num_cluster - 1):
            dist = np.zeros(num_pixel)
            for i in range(num_pixel):
                min_dist = np.Inf
                for j in range(num_center + 1):
                    cur_dist = np.linalg.norm(grid_indices[i] - center[j])
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                dist[i] = min_dist
            dist /= np.sum(dist)
            center.append(grid_indices[np.random.choice(
                num_pixel, 1, p=dist)[0]].tolist())

        return np.array(center)


def initCluster(
        num_row: int,
        num_col: int,
        num_cluster: int,
        kernel: np.ndarray,
        method: int):
    # Choose init center of clusters
    center = chooseCenter(num_row, num_col, num_cluster, method)

    # k-means
    num_pixel = num_row * num_col
    init_cluster = np.zeros(num_pixel, dtype=int)
    # Compute the distance between every point and all centers.
    for i in range(num_pixel):
        dist = np.zeros(num_cluster)
        for j in range(num_cluster):
            center_idx = center[j, 0] * num_row + center[j, 1]
            dist[j] = kernel[i, i] + kernel[center_idx,
                                            center_idx] - 2 * kernel[i, center_idx]
        init_cluster[i] = np.argmin(dist)

    return init_cluster


def setColor(num_cluster: int):
    colors = np.array([[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255]])
    if num_cluster > 3:
        colors = np.append(
            colors, np.random.choice(
                256, (num_cluster - 3, 3)), axis=0)

    return colors


def getCurrentImage(
        num_row: int,
        num_col: int,
        cluster: np.ndarray,
        colors: np.ndarray):
    cur_image = np.zeros((num_row * num_col, 3))
    for i in range(num_row * num_col):
        cur_image[i, :] = colors[cluster[i], :]
    cur_image = cur_image.reshape(num_row, num_col, 3)
    cur_image = Image.fromarray(np.uint8(cur_image))

    return cur_image


def computeKernelKMeans(
        num_row: int,
        num_col: int,
        num_cluster: int,
        cluster: np.ndarray,
        kernel: np.ndarray,
        method: int,
        image_idx: int):
    # Set color for print result
    colors = setColor(num_cluster)

    # Init a image array for save image result
    save_image = [getCurrentImage(num_row, num_col, cluster, colors)]

    num_pixel = num_row * num_col

    iteration = 100
    # Run kernel k-means
    for i in range(1, iteration):
        print(f'iteration {i}')
        prev_cluster = cluster.copy()
        cluster = np.zeros(num_pixel, dtype=int)

        # Get the count array of all cluster
        _, cluster_count = np.unique(prev_cluster, return_counts=True)

        # Compute kernel_pq
        kernel_pq = np.zeros(num_cluster)
        for j in range(num_cluster):
            temp_kernel = kernel.copy()
            for k in range(num_pixel):
                # This pixel not in the same cluster, set to 0
                if prev_cluster[k] != j:
                    temp_kernel[k, :] = 0
                    temp_kernel[:, k] = 0
            # Sum up the pairwise kernel distances of each cluster
            kernel_pq[j] = np.sum(temp_kernel)

        for j in range(num_pixel):
            dist = np.full(num_cluster, np.inf)
            for k in range(num_cluster):
                temp_j = kernel[j, :].copy()
                index_j = np.where(prev_cluster == k)
                kernel_jn = np.sum(temp_j[index_j])

                dist[k] = kernel[j, j] - 2 / cluster_count[k] * \
                    kernel_jn + (1 / cluster_count[k] ** 2) * kernel_pq[k]
            cluster[j] = np.argmin(dist)

        # Save image in image array
        save_image.append(getCurrentImage(num_row, num_col, cluster, colors))

        # Break if cluster is stable.
        if(np.linalg.norm((cluster - prev_cluster), ord=2) < 1e-2):
            break

    # Save gif
    filename_gif = getFilename(num_cluster, method, image_idx, 'gif')
    filename_png = getFilename(num_cluster, method, image_idx, 'png')
    save_image[0].save(filename_gif,
                       save_all=True,
                       append_images=save_image[1:],
                       optimize=False,
                       loop=0,
                       duration=100)
    save_image[-1].save(filename_png)


def getFilename(num_cluster: int, method: int, image_idx: int, ftype: str):
    if ftype == 'gif':
        dirname = './output_images/kernel_kmeans_gif'
    else:
        dirname = './output_images/kernel_kmeans_png'
    if method == 0:
        m = 'Random'
    elif method == 1:
        m = 'Kmeans++'

    filename = f'{dirname}/image_{image_idx}_cluster_{num_cluster}_{m}.{ftype}'
    os.makedirs(dirname, exist_ok=True)

    return filename


def kernelKMeans(
        image: np.ndarray,
        image_idx: int,
        gamma_s: float,
        gamma_c: float,
        num_cluster: int,
        method: int):
    # Get image shape
    num_row, num_col, num_color = image.shape

    # Compute kernel
    kernel = computeKernel(
        image,
        gamma_s,
        gamma_c,
        num_row,
        num_col,
        num_color)

    # Init cluster
    cluster = initCluster(num_row, num_col, num_cluster, kernel, method)

    # Compute Kernel k-means
    computeKernelKMeans(
        num_row,
        num_col,
        num_cluster,
        cluster,
        kernel,
        method,
        image_idx)
