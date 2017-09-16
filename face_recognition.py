from pathlib import Path
from utils import *

#Subjects
SUBJECTS = 40
IMG_PER_SUBJECT = 10
PIXELS_H = 92
PIXELS_V = 112

def get_training_faces(subject):
    c = 0
    A, y = [], []
    images_list = Path("images/"+ subject).glob('**/*.pgm')

    for image in images_list:
        im = np.asarray(Image.open(str(image)).convert('L'))
        A.append(im)
        y.append(c)
        c = c + 1

    return [A, y]


def average_face_vector(face_vectors):
    dim = np.shape(face_vectors)
    average = np.zeros((1, dim[1]))
    for face_vector in face_vectors:
        average += face_vector;
    #average = np.round(average / dim[0]);
    average = average / dim[0]
    return average


def calculate_face_vectors(images):
    dim = np.shape(images)
    face_vectors = np.empty((dim[0], dim[1] * dim[2]))
    for i in range(0, dim[0]):
        face_vectors[i] = matrix_to_vector(images[i])
    return face_vectors

def calculate_eienfaces(eigenvectors):
    eigenfaces = np.empty((IMG_PER_SUBJECT, PIXELS_V, PIXELS_H), dtype="uint8")
    for i in range(0, np.shape(eigenvectors)[0] - 1):
        eigenfaces[i] = vector_to_matix(eigenvectors[i] * 255)

    return eigenfaces

def training_set_gamma_vectors():
    #train_set = np.empty((SUBJECTS, IMG_PER_SUBJECT, PIXELS_V, PIXELS_H), dtype="uint8")
    gamma_vectors_set = np.empty((SUBJECTS, IMG_PER_SUBJECT, IMG_PER_SUBJECT))
    for i in range(0, SUBJECTS - 1):
        [training_faces, y] = get_training_faces("s"+ str(i+1))

        # cada face_vector es de (1, 112*92) = (1, 10304)
        face_vectors = calculate_face_vectors(training_faces)
        average_face_vect = average_face_vector(face_vectors)
        face_vectors_minus_avg = np.matrix(face_vectors - average_face_vect)

        # U = eigenvectors de a * a.H, V = eigenvectors de a.H * a, S = eigenvalues
        # algo estamos haciendo mal en el calculate_eigenvalues, deberia poder calular los de eigenvectors de 10304*10304.
        U, S, V = np.linalg.svd(face_vectors_minus_avg, full_matrices=False)

        eigenfaces = calculate_eienfaces(V)

        gamma_vectors = calculate_gamma_vectors(eigenfaces, face_vectors_minus_avg)
        gamma_vectors_set[i] = gamma_vectors
        #train_set[i] = gamma_vectors

    return gamma_vectors_set

def calculate_gamma_vectors(eigen_faces, face_vectors_minus_avg):
    vector_eigenfaces = np.empty((IMG_PER_SUBJECT, PIXELS_H * PIXELS_V), dtype="uint8")
    for i in range(0, IMG_PER_SUBJECT - 1):
        vector_eigenfaces[i] = matrix_to_vector(eigen_faces[i])

    return np.dot(face_vectors_minus_avg, np.transpose(vector_eigenfaces))



#Aca falta algo, la "eigenface" es un autovector de long 10304. Fierens usa la funcion np.linalg.svd que le da:
#Los autvectores y autovalores q calculamos aca mas los autovectores de long 10304. Esos autovectores son los que termina
#usando. Creo q algo anda mal porque no deberia tardar tanto para calcular esos autovectores.

 # filas < columnas
    #if (face_vectors_minus_avg.shape[0] < face_vectors_minus_avg.H.shape[0]):
    #    S = np.matrix(face_vectors_minus_avg) * np.matrix(face_vectors_minus_avg).H
    # columnas < filas
    #else:
     #   S = np.matrix(face_vectors_minus_avg).H * np.matrix(face_vectors_minus_avg)

    # Antes no estaba ese if, y calculaba la cov de face_vectors_minus_avg.transpose() --> Tabla de 10304*10304
    #covariance_matrix = np.matrix(np.cov(S))
    #eig = calculate_eigenvalues(covariance_matrix)
