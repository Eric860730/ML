import numpy as np
from numba import jit


#python rich for debug
from rich.traceback import install
install(show_locals=True)

NUM_CLASS = 10

@jit
def transformTwoBins(train_image, pixels, num_image):
    two_bins_image = np.zeros((num_image, pixels))
    for i in range(num_image):
        for j in range(pixels):
            if train_image[i, j] > 127:
                two_bins_image[i, j] = 1
    return two_bins_image


@jit
def expectationStep(two_bins_image, P, Lambda, w, pixels, num_image):
    for i in range(num_image):
        for j in range(NUM_CLASS):
            w[i, j] = Lambda[j]
            for k in range(pixels):
                if two_bins_image[i, k]:
                    w[i, j] *= P[j, k]
                else:
                    w[i, j] *= (1 - P[j, k])
        sum_wi = np.sum(w[i, :])
        if sum_wi:
            w[i, :] /= sum_wi
    return w


@jit
def maximizationStep(two_bins_image, P, Lambda, w, pixels, num_image):
    Lambda = np.sum(w, axis = 0)
    for i in range(NUM_CLASS):
        for j in range(pixels):
            P[i, j] = 0
            for k in range(num_image):
                P[i, j] += w[k, i] * two_bins_image[k, j]
            P[i, j] = (P[i, j] + 1e-9) / (Lambda[i] + 1e-9 * 784)
        Lambda[i] = (Lambda[i] + 1e-9) / (np.sum(Lambda) + 1e-9 * 10)
    return P, Lambda



def printImaginations(P, count, distance_P, pixels):
    classify_class = np.zeros((NUM_CLASS, pixels))
    classify_class = (P >= 0.5) * 1
    for class_num in range(NUM_CLASS):
        print(f"class {class_num}:")
        for row in range(28):
            for col in range(28):
                print(classify_class[class_num][row * 28 + col], end = ' ')
            print()
        print()
    print(f"No. of Iteration: {count}, Difference: {distance_P}")
    print("----------------------------------------------------\n")


@jit
def countLabel(two_bins_image, P, Lambda, pixels, num_image, train_label):
    count_matrix = np.zeros((10, 10))
    result = np.zeros(10)

    for i in range(num_image):
        for j in range(NUM_CLASS):
            result[j] = Lambda[j]
            for k in range(pixels):
                if two_bins_image[i, k]:
                    result[j] *= P[j, k]
                else:
                    result[j] *= (1 - P[j, k])

        predict_class = np.argmax(result)
        count[predict_class, train_label[i]] += 1

    return count_matrix


def matchLabel(count_matrix):
    match_matrix = np.full(10, -1, dtype = int)

    for _ in range(10):
        index = np.unravel_index(np.argmax(count_matrix), (10, 10))

        match_matrix[index[0]] = index[1]

        for i in range(10):
            count_matrix[index[0]][i] = -1
            count_matrix[i][index[1]] = -1

    return match_matrix


def predictLabel(match_matrix, two_bins_image, P, Lambda, pixels, num_image, train_label):
    prediction_matrix = np.zeros((10, 10))
    tmp_result = np.zeros(10)

    for i in range(num_image):
        for j in range(NUM_CLASS):
            tmp_result[j] = Lambda[j]
            for k in range(pixels):
                if two_bins_image[i, k]:
                    tmp_result[j] *= P[j, k]
                else:
                    tmp_result[j] *= (1 - P[j, k])

        predict_class = np.argmax(tmp_result)
        prediction_matrix[match_matrix[predict_class], train_label[i]] += 1

    return prediction_matrix

def printResultImagination(prediction_matrix, P, pixels):
    print("\n------------------------------------------------------------\n------------------------------------------------------------\n")

    imagination = (P >= 0.5) * 1
    for i in range(NUM_CLASS):
        index = prediction_matrix[i]

    pass


# 1. Binning the gray level value into two bins. (0: 0~127, 1: 128~256)
# 2. Treating all pixels as random variables following Bernoulli distributions. Given P0, P1, lambda, calculate all singal points.
# 3
def EM_algorithm(train_label, train_image, test_label, test_image):
    pixels = len(train_image[0])
    num_image = len(train_image)
    two_bins_image = transformTwoBins(train_image, pixels, num_image)
    P = np.random.uniform(0.0, 1.0, (NUM_CLASS, pixels))
    for i in range(NUM_CLASS):
        P[i, :] /= np.sum(P[i, :])
    Lambda = np.full(NUM_CLASS, 0.1)
    w = np.zeros((num_image, NUM_CLASS))
    count = 0

    while True:
        previous_P = np.copy(P)
        count += 1
        w = expectationStep(two_bins_image, P, Lambda, w, pixels, num_image)
        P, Lambda = maximizationStep(two_bins_image, P, Lambda, w, pixels, num_image)
        distance_P = np.linalg.norm(P - previous_P)
        printImaginations(P, count, distance_P, pixels)
        if count == 20 or distance_P < 1e-2:
            break

    count_matrix = countLabel(two_bins_image, P, Lambda, pixels, num_image, train_label)
    match_matrix = matchLabel(count_matrix)
    prediction_matrix = predictLabel(match_matrix, two_bins_image, P, Lambda, pixels, num_image, train_label)
    printResultImagination(prediction_matrix, P, pixels)


    print("in EM_algorithm")
    pass
