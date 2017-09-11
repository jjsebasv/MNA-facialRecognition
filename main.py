#!/usr/bin/python3
from face_recognition import *


def main():
    [training_faces, y] = get_training_faces()

    # cada face_vector es de (1, 112*92) = (1, 10304)
    face_vectors = calculate_face_vectors(training_faces)
    average_face_vect = average_face_vector(face_vectors)
    face_vectors_minus_avg = np.matrix(face_vectors - average_face_vect)

    # filas < columnas
    if (face_vectors_minus_avg.shape[0] < face_vectors_minus_avg.H.shape[0]):
        S = np.matrix(face_vectors_minus_avg) * np.matrix(face_vectors_minus_avg).H
    # columnas < filas
    else:
        S = np.matrix(face_vectors_minus_avg).H * np.matrix(face_vectors_minus_avg)

    covariance_matrix = np.matrix(np.cov(S))
    eig = calculate_eigenvalues(covariance_matrix)

    print(eig)
    # pca(as_row_matrix(A))


if __name__ == "__main__":
    main()
