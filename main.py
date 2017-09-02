#!/usr/bin/python3

from utils import get_training_faces
from utils import as_row_matrix
from pca import pca

def main():
    [A, y] = get_training_faces()
    print(y)
    pca(as_row_matrix(A))

if __name__ == "__main__":
    main()
