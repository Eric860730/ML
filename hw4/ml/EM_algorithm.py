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
        # for each image, compute the responsibility(w) of each class
        for j in range(NUM_CLASS):
            # w = λ * p^num_one * (1 - p)^(pixels - num_one), where "num_one" means the number of 1 in image.
            w[i, j] = Lambda[j]
            for k in range(pixels):
                if two_bins_image[i, k]:
                    w[i, j] *= P[j, k]
                else:
                    w[i, j] *= (1 - P[j, k])
        # Normalized
        sum_wi = np.sum(w[i, :])
        if sum_wi:
            w[i, :] /= sum_wi
    return w


@jit
def maximizationStep(two_bins_image, P, Lambda, w, pixels, num_image):
    # Get sum of w
    sum_w = np.sum(w, axis = 0)
    for i in range(NUM_CLASS):
        for j in range(pixels):
            # Initialize P[i, j] to 0
            P[i, j] = 0
            # p = Σ(w * xi) + 1e-9 / (Σw + 1e-9 * pixels)
            for k in range(num_image):
                P[i, j] += w[k, i] * two_bins_image[k, j]
            P[i, j] = (P[i, j] + 1e-9) / (sum_w[i] + 1e-9 * pixels)
        # Lambda = Normalized (Σw + 1e-9)
        Lambda[i] = (sum_w[i] + 1e-9) / (np.sum(sum_w) + 1e-9 * NUM_CLASS)
    return P, Lambda


def printImaginations(P, count, distance_P, pixels):
    classify_class = np.zeros((NUM_CLASS, pixels))
    classify_class = (P >= 0.5) * 1
    for i in range(NUM_CLASS):
        print(f"class {i}:")
        for row in range(28):
            for col in range(28):
                print(classify_class[i][row * 28 + col], end = ' ')
            print()
        print()
    print(f"No. of Iteration: {count}, Difference: {distance_P}")
    print("----------------------------------------------------\n")


# count our predict class and corresponding real class and record it in a count_matrix
@jit
def countLabel(two_bins_image, P, Lambda, pixels, num_image, train_label):
    count_matrix = np.zeros((10, 10))
    result = np.zeros(10)

    for i in range(num_image):
        # for each image, calculate the probability of each class and store in result.
        for j in range(NUM_CLASS):
            result[j] = Lambda[j]
            for k in range(pixels):
                if two_bins_image[i, k]:
                    result[j] *= P[j, k]
                else:
                    result[j] *= (1 - P[j, k])

        # select largest probability of result and count_matrix[row][col] += 1, where row is our largest probability of result, col is the true class number.
        predict_class = np.argmax(result)
        count_matrix[predict_class, train_label[i]] += 1

    return count_matrix


# According to count_matrix, match our class to the real class.
@jit
def matchLabel(count_matrix):
    match_array = np.full(10, -1, dtype = int)

    for _ in range(10):
        # select the index of the max value in count_matrix, index = (row, col)
        index = np.unravel_index(np.argmax(count_matrix), (10, 10))

        # match our class to real class
        # e.g. index = (3, 5) means match_array[3] = 5, i.e. our class 3 is real class 5
        match_array[index[0]] = index[1]

        # set all the index's row and col to -1, avoid repeat match.
        for i in range(10):
            count_matrix[index[0]][i] = -1
            count_matrix[i][index[1]] = -1

    return match_array


# According to match_array, calculate our final prediction matrix.
@jit
def predictLabel(match_array, two_bins_image, P, Lambda, pixels, num_image, train_label):
    prediction_matrix = np.zeros((10, 10))

    # tmp result of each class of per image
    tmp_result = np.zeros(10)

    for i in range(num_image):
        for j in range(NUM_CLASS):
            tmp_result[j] = Lambda[j]
            for k in range(pixels):
                if two_bins_image[i, k]:
                    tmp_result[j] *= P[j, k]
                else:
                    tmp_result[j] *= (1 - P[j, k])

        # set max value as our predict_class
        predict_class = np.argmax(tmp_result)

        # construct prediction_matrix, row index means our predict result of image, col index means the real result of image.
        # e.g. prediction_matrix[1][2] means we classify the image to class 1 but in fact the image is class 2.
        prediction_matrix[match_array[predict_class], train_label[i]] += 1

    return prediction_matrix


def printResultImagination(match_array, P, pixels):
    print("\n------------------------------------------------------------\n------------------------------------------------------------\n")

    imagination = (P >= 0.5) * 1
    for i in range(NUM_CLASS):
        result_class = match_array[i]
        print(f"labeled class {i}:")
        for row in range(28):
            for col in range(27):
                print(f"{imagination[result_class][row * col + col]}", end = " ")
            print(f"{imagination[result_class][row * 27 + 27]}")
        print()
    print()


def printConfusionMatrix(prediction_matrix, count, num_image):
    error_count = num_image

    for i in range(NUM_CLASS):
        tp, fp, tn, fn = computeConfusionMatrix(i, prediction_matrix)
        error_count -= tp
        print("\n-------------------------------------------------------\n")
        print(f"Confusion Matrix {i}:")
        print(f"                  Predict number {i} Predict not number {i}")
        print(f"Is number {i}           {tp:5>}              {fn:5>}")
        print(f"Isn't number {i}        {fp:5>}              {tn:5>}")
        print(f"\nSensitivity (Successfully predict number {i}    : {float(tp) / (tp + fn):.5f})")
        print(f"Specificity (Successfully predict not number {i}: {float(tn) / (fp + tn):.5f})")

    print(f"\nTotal iteration to converge: {count}")
    print(f"Total error rate: {float(error_count) / num_image:.16f}")


# compute Confusion matrix
# prediction_matrix
#   row_num: predict class
#   col_num: real class
@jit
def computeConfusionMatrix(class_num, prediction_matrix):
    tp, fp, tn, fn = 0, 0, 0, 0
    for prediction in range(10):
        for answer in range(10):
            if prediction == class_num and answer == class_num:
                tp += prediction_matrix[prediction, answer]
            elif prediction == class_num:
                fp += prediction_matrix[prediction, answer]
            elif answer == class_num:
                fn += prediction_matrix[prediction, answer]
            else:
                tn += prediction_matrix[prediction, answer]

    return tp, fp, fn, tn


# EM_algorithm overall
# 1. Binning the gray level value into two bins. (0: 0~127, 1: 128~256)
# 2. Treating all pixels as random variables following Bernoulli distributions.
#   Given P1 ~ P10(Record the probability of occurrence of each pixel on the graph in each category)
#   Lambda(Probability of occurrence in each category), calculate all singal points.
# 3. Calculate Expectation Step (E-step) to get w, which record probability of each category of every train_image.
# 4. Use w get from E-step, calculate Maximization Step (M-step) to get MLE of Lambda and P.
# 5. Repeat 3 and 4 until MLE of P converge or count == 20
# 6. Visualize result.
def EM_algorithm(train_label, train_image, test_label, test_image):
    # total pixels in each train_image
    pixels = len(train_image[0])

    # total number of train_image
    num_image = len(train_image)

    # Convert train_image to two_bins_image
    two_bins_image = transformTwoBins(train_image, pixels, num_image)

    # Given arbitrary P and normalized
    P = np.random.uniform(0.0, 1.0, (NUM_CLASS, pixels))
    for i in range(NUM_CLASS):
        P[i, :] /= np.sum(P[i, :])

    # Given arbitrary Lambda(set all Lambda to 0.1)
    Lambda = np.full(NUM_CLASS, 0.1)

    # init w to 0
    w = np.zeros((num_image, NUM_CLASS))
    count = 0

    while True:
        previous_P = np.copy(P)
        count += 1

        # E-step, get new W
        w = expectationStep(two_bins_image, P, Lambda, w, pixels, num_image)

        # M-step, get new P, Lambda
        P, Lambda = maximizationStep(two_bins_image, P, Lambda, w, pixels, num_image)

        # calculate distance between new P and previous P
        distance_P = np.linalg.norm(P - previous_P)

        # print current P in each EM finished
        printImaginations(P, count, distance_P, pixels)

        # break conditional
        if count == 20 or distance_P < 1e-2:
            break

    # According final probability P, record finally category result and corresponding with answer category.
    count_matrix = countLabel(two_bins_image, P, Lambda, pixels, num_image, train_label)

    # Match our classified result to real class.
    match_array = matchLabel(count_matrix)

    # Get final preduction result matrix
    prediction_matrix = predictLabel(match_array, two_bins_image, P, Lambda, pixels, num_image, train_label)

    # Print result imagination of each class
    printResultImagination(match_array, P, pixels)

    # Print Confusion Matrix
    printConfusionMatrix(prediction_matrix, count, num_image)


