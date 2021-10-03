import sys
import ml


def main():
    if (len(sys.argv) != 2):
        print("Please Enter mode.")
        exit()
    train_label, train_image, test_label, test_image = ml.read_data('./hw2_data/')
    ml.calculateNaiveBayesClassifier(train_label, train_image, test_label, test_image, int(sys.argv[1]))


if __name__ == '__main__':
    main()
