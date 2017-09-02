import numpy as np
from utils import calculate_eigenvalues

def covariance_matrix(A):
    return np.cov(np.vstack([A]))

def pca(A):
    print('Running pca\n')

    cov_matrix = covariance_matrix(A)
    print('Cov Matrix')
    print(cov_matrix)
    print('\n')

    eigenvalues = calculate_eigenvalues(np.asmatrix(cov_matrix))
    print('Eigenvalues')
    print(eigenvalues)
    print('\n')
