import numpy as np

CLASS_NUM = 10

def calculatePrior(train_label):
    prior = np.zeros(CLASS_NUM, dtype=float)
    train_label_num = len(train_label)
    for i in range(train_label_num):
        prior[train_label[i]] = prior[train_label[i]] + 1
    for i in range(len(prior)):
        prior[i] = prior[i] / train_label_num

    return prior


#def calculateLikelihood():

def calculateNaiveBayesClassifier(train_label, train_image, test_label, test_image):
    prior = calculatePrior(train_label)
    print(prior)

