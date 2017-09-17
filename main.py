#!/usr/bin/python3
from face_recognition import *

def main():
    clf, avg_image, V = training_set_gamma_vectors();
    acc = 0
    total = 0
    for i in range(1, 6):
        for j in range(0, 4):
            total += 1
            ans = parse_query(i, j, clf, avg_image, V)
            print(ans, i)
            if ans == i:
                acc += 1

    print(1.0*acc/total)


if __name__ == "__main__":
    main()
