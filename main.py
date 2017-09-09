#!/usr/bin/python3
from face_recognition import *


def main():
    [training_faces, y] = get_training_faces()

    # cada face_vector es de (1, 112*92) = (1, 10304)
    face_vectors = calculate_face_vectors(training_faces)
    average_face_vect = average_face_vector(face_vectors)

    face_vectors_minus_avg = face_vectors - average_face_vect
    covariance_matrix = np.cov(face_vectors_minus_avg.transpose())

    # eig = calculate_eigenvalues(covariance_matrix)
    # pca(as_row_matrix(A))


if __name__ == "__main__":
    main()
