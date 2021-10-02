import sys
import ml


def main():
    train_label, train_image, test_label, test_image = ml.read_data('./hw2_data/')
    ml.calculateNaiveBayesClassifier(train_label, train_image, test_label, test_image)


if __name__ == '__main__':
    main()
