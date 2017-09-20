from pathlib import Path

from utils import *

# Subjects

PIXELS_H = 92
PIXELS_V = 112


class PCAQueryParams():
    training_projections = None
    test_projections = None
    average_image = None
    eigenvectors = None


class KernelPCAQueryParams(PCAQueryParams):
    K = None
    K_test = None
    observations = None
    images = None
    degree = None
    ones = None


def get_training_faces_for_subject(args, subject):
    A, y = [], []
    path = "%s/%s" % (args.imgdb, subject)
    images_list = Path(path).glob('**/*.pgm')

    for i, image in enumerate(images_list):
        if i >= args.img_per_subject:
            break
        im = np.asarray(Image.open(str(image)).convert('L'))
        # im = (np.asarray(Image.open(str(image)).convert('L')) - 127.5) / 127.5
        A.append(im)
        y.append(i)

    return [A, y]


def get_test_faces(args, subject):
    A, y = [], []
    path = "%s/%s" % (args.imgdb, subject)
    images_list = Path(path).glob('**/*.pgm')

    for i, image in enumerate(images_list):
        if i >= args.img_per_subject:
            im = np.asarray(Image.open(str(image)).convert('L'))
            # im = (np.asarray(Image.open(str(image)).convert('L')) - 127.5) / 127.5
            A.append(im)
            y.append(i)

    return [A, y]


def average_face_vector(face_vectors):
    dim = np.shape(face_vectors)
    average = np.zeros((1, dim[1]))
    for face_vector in face_vectors:
        average += face_vector
    # average = np.round(average / dim[0]);
    average = average / dim[0]
    return average


def calculate_face_vectors(images):
    dim = np.shape(images)
    face_vectors = np.empty((dim[0], dim[1] * dim[2]))
    for i in range(0, dim[0]):
        face_vectors[i] = matrix_to_vector(images[i])
    return face_vectors


def calculate_eienfaces(args, eigenvectors):
    eigenfaces = np.empty((args.img_per_subject, PIXELS_V, PIXELS_H), dtype="uint8")
    for i in range(0, np.shape(eigenvectors)[0]):
        eigenfaces[i] = vector_to_matix(eigenvectors[i] * 255)

    return eigenfaces


def get_training_images(args):
    images = np.empty((args.subjects, args.img_per_subject, PIXELS_H * PIXELS_V))
    for i in range(0, args.subjects):
        [training_faces, y] = get_training_faces_for_subject(args, "s%d" % (i + 1))

        # cada face_vector es de (1, 112*92) = (1, 10304)
        face_vectors = calculate_face_vectors(training_faces)
        images[i] = face_vectors

    return images.reshape((args.subjects * args.img_per_subject, PIXELS_H * PIXELS_V))


def get_test_images(args):
    test_images = np.empty((args.subjects, args.test_img_per_subject, PIXELS_H * PIXELS_V))
    for i in range(0, args.subjects):
        [test_faces, y] = get_test_faces(args, "s%d" % (i + 1))

        # cada face_vector es de (1, 112*92) = (1, 10304)
        test_face_vectors = calculate_face_vectors(test_faces)
        test_images[i] = test_face_vectors

    return test_images.reshape((args.subjects * args.test_img_per_subject, PIXELS_H * PIXELS_V))


def pca(args):
    images = get_training_images(args)

    # Restamos la media a todas las imagenes de entrenamiento
    average_face_vect = average_face_vector(images)
    face_vectors_minus_avg = np.empty((images.shape[0], PIXELS_H * PIXELS_V))
    for i in range(images.shape[0]):
        face_vectors_minus_avg[i] = images[i] - average_face_vect

    test_images = get_test_images(args)

    # Restamos la media a todas las imagenes de testing
    test_face_vectors_minus_avg = np.empty((test_images.shape[0], PIXELS_H * PIXELS_V))
    for i in range(test_images.shape[0]):
        test_face_vectors_minus_avg[i] = test_images[i] - average_face_vect

    # Calculamos los autovectores de A'A o AA' dependiendo de las dimensiones
    if len(face_vectors_minus_avg) > face_vectors_minus_avg[0].shape[0]:
        Ss = np.matrix(face_vectors_minus_avg).H * np.matrix(face_vectors_minus_avg)
        eigenvalues, eigenvectors = calculate_eigenvalues(Ss)
    # columnas > filas
    else:
        Ss = np.matrix(face_vectors_minus_avg) * np.matrix(face_vectors_minus_avg).H
        eigenvalues, eigenvectors = calculate_eigenvalues(Ss)
        eigenvectors = face_vectors_minus_avg.transpose() * eigenvectors
        for i in range(face_vectors_minus_avg.shape[0]):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    #
    # idx = np.argsort(-eigenvalues)
    # eigenvalues = eigenvalues[idx]
    # eigenvectors = eigenvectors[:,idx]
    # eigenvectors = np.reshape(eigenvectors,eigenvectors.shape[0:2])

    # U = eigenvectors de a * a.H, V = eigenvectors de a.H * a, S = eigenvalues
    # algo estamos haciendo mal en el calculate_eigenvalues, deberia poder calular los de eigenvectors de 10304*10304.


    # Esta es la magia que no tenemos que borrar por las dudas
    # U, S, V = np.linalg.svd(face_vectors_minus_avg, full_matrices=False)

    # B = V[0:V.shape[0],:]
    # proyecto

    improy = np.dot(face_vectors_minus_avg, eigenvectors)
    imtstproy = np.dot(test_face_vectors_minus_avg, eigenvectors)

    query_params = PCAQueryParams()
    query_params.average_image = average_face_vect
    query_params.eigenvectors = eigenvectors
    query_params.training_projections = improy
    query_params.test_projections = imtstproy

    return query_params

    # # SVM
    # # entreno
    # clf = svm.LinearSVC()
    # clf.fit(improy, person.ravel())
    # accs = clf.score(imtstproy, persontst.ravel())
    # # print("Precision "+ str(accs * 100) +"\n")
    #
    # return clf, average_face_vect, eigenvectors


def kpca(args):
    images = get_training_images(args)
    test_images = get_test_images(args)

    observations = args.subjects * args.img_per_subject

    degree = 2
    K = (np.dot(images, images.T) / observations + 1) ** degree  # Ecuacion 61 del paper

    # Centramos las observaciones
    # Ecuacion 49 del paper http://www.face-rec.org/algorithms/Kernel/kernelPCA_scholkopf.pdf
    ones = np.ones([observations, observations]) / observations
    K_dot_ones = np.dot(K, ones)
    K = K - np.dot(ones, K) - K_dot_ones + np.dot(ones, K_dot_ones)

    eigenvalues, eigenvectors = calculate_eigenvalues(np.matrix(K))

    # Los autovalores vienen en orden descendente. Lo cambio
    # Tomado del archivo de ejemplo en el campus
    eigenvalues = np.flipud(eigenvalues)
    eigenvectors = np.fliplr(eigenvectors)

    for col in range(eigenvectors.shape[1]):
        # FIXME Corregido con np.abs
        eigenvectors[:, col] = eigenvectors[:, col] / np.sqrt(eigenvalues[col])  # Normalizacion. Sec B.2

    #test_cases = args.subjects * args.test_img_per_subject
    #ones_test = np.ones([test_cases, observations]) / observations

    # Ecuacion 52 del paper
    #K_test = (np.dot(test_images, images.T) / observations + 1) ** degree
    # Ecuacion 54 del paper
    #ones_test_dot_K = np.dot(ones_test, K)
    #K_test = K_test - ones_test_dot_K - np.dot(K_test, ones) + np.dot(ones_test_dot_K, ones)

    query_params = KernelPCAQueryParams()
    query_params.eigenvectors = eigenvectors
    query_params.training_projections = np.dot(K.T, eigenvectors)  # Ecuacion 17
    # query_params.test_projections = np.dot(K_test, eigenvectors)  # Ecuacion 51
    # query_params.K_test = K_test
    query_params.images = images
    query_params.degree = degree
    query_params.ones = ones
    query_params.K = K
    query_params.observations = observations

    return query_params


def pca_query(args, subject, image, clf, query_params: PCAQueryParams):
    image = np.array(matrix_to_vector(get_test_faces(args, "s%s" % subject)[0][image]))
    diff = image - query_params.average_image
    improy = np.dot([diff], query_params.eigenvectors)
    return clf.predict(improy)


def kpca_query(args, subject, image, clf, query_params: KernelPCAQueryParams):
    image = np.array(matrix_to_vector(get_test_faces(args, "s%s" % subject)[0][image]))
    # improy = np.dot([image], query_params.eigenvectors)

    test_cases = 1
    ones_test = np.ones([test_cases, query_params.observations]) / query_params.observations

    # Ecuacion 52 del paper
    K_test = (np.dot([image], query_params.images.T) / query_params.observations + 1) ** query_params.degree
    # Ecuacion 54 del paper
    ones_test_dot_K = np.dot(ones_test, query_params.K)
    K_test = K_test - ones_test_dot_K - np.dot(K_test, query_params.ones) + np.dot(ones_test_dot_K, query_params.ones)

    imtstproypre = np.dot(K_test, query_params.eigenvectors)  # Ecuacion 51

    return clf.predict(imtstproypre)

# Aca falta algo, la "eigenface" es un autovector de long 10304. Fierens usa la funcion np.linalg.svd que le da:
# Los autvectores y autovalores q calculamos aca mas los autovectores de long 10304. Esos autovectores son los que termina
# usando. Creo q algo anda mal porque no deberia tardar tanto para calcular esos autovectores.

# filas < columnas
# if (face_vectors_minus_avg.shape[0] < face_vectors_minus_avg.H.shape[0]):
#    S = np.matrix(face_vectors_minus_avg) * np.matrix(face_vectors_minus_avg).H
# columnas < filas
# else:
#   S = np.matrix(face_vectors_minus_avg).H * np.matrix(face_vectors_minus_avg)

# Antes no estaba ese if, y calculaba la cov de face_vectors_minus_avg.transpose() --> Tabla de 10304*10304
# covariance_matrix = np.matrix(np.cov(S))
# eig = calculate_eigenvalues(covariance_matrix)
