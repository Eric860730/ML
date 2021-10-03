import sys
import ml
import os.path


def usage():
    print("Usage: python3 hw2-1 <mode>")
    print("Usage: python3 hw2-2 <a> <b>")
    exit()


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        usage()

    if (sys.argv[1] == 'hw2-1'):
        if (len(sys.argv) != 3):
            usage()
        train_label, train_image, test_label, test_image = ml.read_data(
            './hw2_data/')
        ml.calculateNaiveBayesClassifier(
            train_label, train_image, test_label, test_image, int(
                sys.argv[2]))

    elif (sys.argv[1] == 'hw2-2'):
        if (len(sys.argv) != 4):
            usage()
        if (not (os.path.exists('hw2_data/testfile.txt'))):
            print(
                "Can not find testfile.txt in hw2_data!\nPlease put testfile.txt in hw2_data and try again.")
            exit()
        testfile = os.path.join('hw2_data', 'testfile.txt')
        ml.onlineLearning(int(sys.argv[2]), int(sys.argv[3]), testfile)

    else:
        usage()
