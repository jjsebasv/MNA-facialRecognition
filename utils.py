from PIL import Image
import numpy as np


def matrix_to_vector(matrix):
    """
    Given a matrix returns a flatten vector of it.

    Keyword arguments:
    matrix -- the matrix to be reshapen
    """
    return (np.reshape(matrix, 92 * 112).astype(np.uint8) - 127.5) / 127.5


def vector_to_matix(face_vector):
    """
    Given a vector returns a matrix of it.

    Keyword arguments:
    face_vector -- the vector to be reshapen
    """
    return face_vector.reshape((112, 92)).astype(np.uint8)


def show_image(image_matrix):
    """
    Given a matrix representing an image, it shows it.

    Keyword arguments:
    image_matrix -- the matrix representation of an image
    """
    img = Image.fromarray(image_matrix, 'L')
    img.show()


def calculate_hessenberg(A):
    """
    Calculates the Hesenberg factorization of a given matrix and returns it.

    Keyword arguments:
    A -- the matrix to which calculate de Hessenberg fatorization
    """
    m = A.shape[0]  # m --> rows
    H = np.matrix(A)
    for j in range(0, m - 2):
        aux = range(j + 1, m)
        x = H[np.ix_(aux, [j])]
        x[0] = x[0] + np.sign(x[0]) * np.linalg.norm(x)
        n = np.linalg.norm(x)
        if n > 0:
            u = x / np.linalg.norm(x)
            aux2 = range(j, m)
            Hh = H[np.ix_(aux, aux2)] - 2 * u * (u.H * H[np.ix_(aux, aux2)])
            for r in aux:
                for c in aux2:
                    H[r, c] = Hh[r - j - 1, c - j]
            aux3 = range(0, m)
            Hh = H[np.ix_(aux3, aux)] - 2 * (H[np.ix_(aux3, aux)] * u) * u.T
            for r in aux3:
                for c in aux:
                    H[r, c] = Hh[r, c - 1 - j]

    return H

def diag(n):
    """
    Returns a square matrix of n rows with 1 on the diag and 0 on the rest.

    Keyword arguments:
    n -- matrix size
    """
    D = np.zeros((n, n))
    for i in range(0, n):
        D[i, i] = 1
    return D

def calculate_givensSinCos(a, b):
    """
    [Auxiliar to calculate_givensQR]

    Calculates the Givens sin and cos over two given elements of a matrix.
    Returns the converted numbers.

    Keyword arguments:
    a, b -- Two matrix numbers to be compared and factorized
    """
    if b != 0:
        if np.absolute(b) > np.absolute(a):
            r = a / b
            s = 1 / np.sqrt(1 + np.power(r, 2))
            c = s * r
        else:
            r = b / a
            c = 1 / np.sqrt(1 + np.power(r, 2))
            s = c * r
    else:
        c = 1
        s = 0
    return c, s


def calculate_givensQR(A):
    """
    Calculates the Givens rotation and returns the first QR iteration

    Keyword arguments:
    A -- the matrix to which calculate de Givens fatorization
    """
    n = A.shape[0]  # n --> rows
    m = A.H.shape[0]  # m --> cols
    Q = np.matrix(diag(n))
    R = A
    for j in range(0, m):
        for i in range(n - 1, j, -1):
            G = diag(n)
            (c, s) = calculate_givensSinCos(R[i - 1, j], R[i, j])
            G[i - 1, i - 1] = c
            G[i - 1, i] = -s
            G[i, i - 1] = s
            G[i, i] = c
            Q = Q * G
            R = Q.H * A
    return (Q, R)

def check_eps(A, oldA):
    """
    Checks if from one iteration to anothe, the matrix had already converged
    with an error lower than eps

    Keyword arguments:
    A -- The new resulting matrix
    oldA -- The last iteration of A
    """
    eps = 0.000001
    for i in range(0, A.shape[0]):
        if abs(A[i,i] - oldA[i,i]) > eps:
            return False
    return True

def iterate_QR(A, Q, R):
    """
    Iterates over the QR given until max_iterations and returns QR

    Keyword arguments:
    A -- The original matrix to check the floating point
    QR -- the QR descomposition to which iterate
    """
    max_iteractions = 100
    flag = False
    eigvectors = Q

    for i in range(0, max_iteractions):
        print(".", end="")
        if not flag:
            AUX = Q * R
        else:
            AUX = R * Q
        flag = not flag

        if i%2 == 1 and check_eps(AUX, A):
            break

        A = AUX
        Q, R = calculate_givensQR(A)
        eigvectors = eigvectors * Q
    print()
    return np.matrix(A), np.matrix(eigvectors)


def get_eigenvalues(A):
    """
    Given a matrix with eigenvalues on the diag, returns an array with
    the eigenvalues normalized (if eigenvalue < convergence ==> eigenvalue = 0)

    Keyword arguments:
    A -- the matrix with eigenvalues on the diag
    """
    eigenvalues = []
    convergence = 0.0000000000000000000001

    for i in range(0, A.shape[0]):
        value = A.item((i, i)) if abs(A.item((i, i))) > convergence else 0
        eigenvalues.append(value)

    return eigenvalues


def calculate_eigenvalues(A):
    """
    Recieves a square matrix and calculate eigenvalues and eigenvectors.
    The algorithm first calculate the Hessenberg matrix of the given one,
    then it calculates the givens factorization.

    Keyword arguments:
    A -- the matrix to which calculate eigenvalues and eigenvectors
    """
    if A.shape[0] == A.H.shape[0]:
        H = calculate_hessenberg(A)
        (Q, R) = calculate_givensQR(H)
        eigenvalues, eigenvectors = iterate_QR(H, Q, R)

        return get_eigenvalues(eigenvalues), eigenvectors
    else:
        print("Error: You should provide a square matrix")
