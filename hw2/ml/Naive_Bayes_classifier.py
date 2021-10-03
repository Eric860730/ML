import numpy as np

# python rich for debug
from rich.traceback import install
install(show_locals=True)

CLASS_NUM = 10
BIN_VALUE_RANGE = 32


def calculatePrior(train_label):
    prior = np.zeros(CLASS_NUM, dtype=float)
    train_label_num = len(train_label)
    for i in range(train_label_num):
        prior[train_label[i]] = prior[train_label[i]] + 1
    for i in range(len(prior)):
        prior[i] = prior[i] / train_label_num

    return prior


def calculateLikelihood(train_label, train_image):
    pixel = len(train_image[0])
    likelihood = np.zeros((CLASS_NUM, pixel, BIN_VALUE_RANGE), dtype=float)
    for i in range(len(train_image)):
        label = train_label[i]
        sum_likelihood = 0
        for j in range(pixel):
            # compress bin_value from 256 to 32
            bin_value = train_image[i, j] // 8
            likelihood[label, j, bin_value] += 1

    total_likelihood = np.sum(likelihood, axis=2)
    # find min_likelihood
    min_likelihood = 60000
    for i in range(CLASS_NUM):
        for j in range(pixel):
            for k in range(BIN_VALUE_RANGE):
                if(likelihood[i, j, k] != 0):
                    if(likelihood[i, j, k] < min_likelihood):
                        min_likelihood = likelihood[i, j, k]

    # Let pseudocount equal min_likelihood
    for i in range(CLASS_NUM):
        for j in range(pixel):
            for k in range(BIN_VALUE_RANGE):
                if likelihood[i][j][k] == 0:
                    likelihood[i][j][k] = min_likelihood

    for i in range(CLASS_NUM):
        for j in range(pixel):
            likelihood[i, j, :] /= total_likelihood[i, j]
    return likelihood


def predictInDiscreteMode(prior, likelihood, test_label, test_image):
    error_num = 0
    pixel = len(test_image[1])
    test_image_num = len(test_image)
    for i in range(test_image_num):
        # calculate posterior
        posterior = np.zeros(CLASS_NUM, dtype=float)
        for j in range(CLASS_NUM):
            for k in range(pixel):
                bin_value = test_image[i][k] // 8
                posterior[j] += np.log(likelihood[j][k][bin_value])
            posterior[j] += np.log(prior[j])

        posterior_sum = np.sum(posterior)
        posterior[:] = posterior[:] / posterior_sum
        print("Postirior (in log scale):")
        for k in range(CLASS_NUM):
            print(f"{k}: {posterior[k]}")
        predict = np.argmin(posterior)
        print(f"Prediction: {predict}, Ans: {test_label[i]}\n")

        if(predict != test_label[i]):
            error_num += 1
    error_rate = error_num / test_image_num
    return error_rate


def showDiscreteImagination(likelihood):
    zero = np.sum(likelihood[:, :, 0:16], axis=2)
    one = np.sum(likelihood[:, :, 16:32], axis=2)
    pixel = len(likelihood[1])
    imagination = np.zeros((CLASS_NUM, pixel), dtype = int)
    for i in range(CLASS_NUM):
        for j in range(pixel):
            if(one[i][j] > zero[i][j]):
                imagination[i][j] = 1

    print("Imagination of numbers in Bayesian classifier:\n")
    row = int(pixel ** 0.5)
    col = row
    for i in range(CLASS_NUM):
        print(f"{i}:")
        for j in range(row):
            for k in range(col):
                print(f"{imagination[i][j*row + k]} ", end='')
            print()
        print()


def calculateNaiveBayesClassifier(
        train_label,
        train_image,
        test_label,
        test_image):
    #prior = calculatePrior(train_label)
    #likelihood = calculateLikelihood(train_label, train_image)
    #np.save("prior.npy", prior)
    #np.save("likelihood.npy", likelihood)
    prior = np.load("prior.npy")
    likelihood = np.load("likelihood.npy")
    error_rate = predictInDiscreteMode(
        prior, likelihood, test_label, test_image)
    showDiscreteImagination(likelihood)
    print(f"Error rate: {error_rate}")
    exit()
