import sys
import ml

def main():
    ml.R_Regression()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Invalid argument number!\nUsage:\n$ python3 main.py <csvfile> <n> <lambda>")
        exit()
    main()
