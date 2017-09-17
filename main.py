#!/usr/bin/python3
from face_recognition import *

def main():
    clf, avg_image, V = training_set_gamma_vectors();
    parse_query(2, 2, clf, avg_image, V)


if __name__ == "__main__":
    main()
