#!/usr/bin/python3
from face_recognition import *

def main():
    clf, avg_image, V = training_set_gamma_vectors();
    # Persona, Foto
    parse_query(2, 1, clf, avg_image, V)


if __name__ == "__main__":
    main()
