from pathlib import Path
from utils import *
from sklearn import svm

#Subjects
SUBJECTS = 10
IMG_PER_SUBJECT = 7
TEST_IMG_PER_SUBJECT = 3
PIXELS_H = 92
PIXELS_V = 112

def get_training_faces(subject):
    c = 0
    A, y = [], []
    images_list = Path("images/"+ subject).glob('**/*.pgm')

    for image in images_list:
        if c > 6:
            break
        im = np.asarray(Image.open(str(image)).convert('L'))
        A.append(im)
        y.append(c)
        c = c + 1

    return [A, y]

def get_test_faces(subject):
    c = 0
    A, y = [], []
    images_list = Path("images/"+ subject).glob('**/*.pgm')

    for image in images_list:
        if c > 6:
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
    images = np.empty((SUBJECTS, IMG_PER_SUBJECT, PIXELS_H * PIXELS_V))
    for i in range(0, SUBJECTS):
        [training_faces, y] = get_training_faces("s"+ str(i+1))

        # cada face_vector es de (1, 112*92) = (1, 10304)
        face_vectors = calculate_face_vectors(training_faces)
        images[i] = face_vectors

    images = images.reshape((SUBJECTS * IMG_PER_SUBJECT, PIXELS_H * PIXELS_V))
    average_face_vect = average_face_vector(images)
    face_vectors_minus_avg = np.empty((images.shape[0], PIXELS_H * PIXELS_V))
    for i in range(images.shape[0]):
        face_vectors_minus_avg[i] = images[i] - average_face_vect
    print(face_vectors_minus_avg[0].shape)

    test_images = np.empty((SUBJECTS, TEST_IMG_PER_SUBJECT, PIXELS_H * PIXELS_V))
    for i in range(0, SUBJECTS):
        [test_faces, y] = get_test_faces("s"+ str(i+1))

        # cada face_vector es de (1, 112*92) = (1, 10304)
        test_face_vectors = calculate_face_vectors(test_faces)
        test_images[i] = test_face_vectors

    test_images = test_images.reshape((SUBJECTS * TEST_IMG_PER_SUBJECT, PIXELS_H * PIXELS_V))
    test_face_vectors_minus_avg = np.matrix(test_face_vectors - average_face_vect)

    if (len(face_vectors_minus_avg) > face_vectors_minus_avg[0].shape[0]):
        Ss = np.matrix(face_vectors_minus_avg).H * np.matrix(face_vectors_minus_avg)
        (eigenvectors, eigenvalues) = calculate_eigenvalues(Ss)
        print('primer if')
    # columnas > filas
    else:
        Ss = np.matrix(face_vectors_minus_avg) * np.matrix(face_vectors_minus_avg).H
        print(Ss.shape)
        (eigenvectors, eigenvalues) = calculate_eigenvalues(Ss)
        print(eigenvalues)
        eigenvectors = face_vectors_minus_avg.H * eigenvectors
        for i in range(face_vectors_minus_avg.shape[0]):
            print(i)
            eigenvectors[:,i] = eigenvectors[:,i] / np.linalg.norm(eigenvectors[:,i])

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    eigenvectors = np.reshape(eigenvectors,eigenvectors.shape[0:2])

    # U = eigenvectors de a * a.H, V = eigenvectors de a.H * a, S = eigenvalues
    # algo estamos haciendo mal en el calculate_eigenvalues, deberia poder calular los de eigenvectors de 10304*10304.
    U, S, V = np.linalg.svd(face_vectors_minus_avg, full_matrices=False)
    # B = V[0:V.shape[0],:]
    #proyecto
    improy      = np.dot(images,np.transpose(V))
    imtstproy   = np.dot(test_images,np.transpose(V))
    person      = np.array([[i + 1] * IMG_PER_SUBJECT for i in range(SUBJECTS)])
    persontst   = np.array([[i + 1] * TEST_IMG_PER_SUBJECT for i in range(SUBJECTS)])

    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    accs = clf.score(imtstproy,persontst.ravel())
    print('Precision con {0} autocaras: {1} %\n'.format(100,accs*100))

    return clf

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
