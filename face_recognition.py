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
    """
    Given a subject get his training faces

    Keyword arguments:
    subject -- given subject
    args --
            img_per_subject
            imgdb
            method
            subjects
            test_img_per_subject
    """

    A = []
    path = "%s/%s" % (args.imgdb, subject)
    images_list = Path(path).glob('**/*.pgm')

    for i, image in enumerate(images_list):
        if i >= args.img_per_subject:
            break
        im = np.asarray(Image.open(str(image)).convert('L'))
        A.append(im)

    return A


def get_test_faces_for_subject(args, subject):
    """
    Given a subject get his test faces

    Keyword arguments:
    subject -- given subject
    args --
            img_per_subject
            imgdb
            method
            subjects
            test_img_per_subject
    """

    A = []
    path = "%s/%s" % (args.imgdb, subject)
    images_list = Path(path).glob('**/*.pgm')

    for i, image in enumerate(images_list):
        if i >= args.img_per_subject:
            im = np.asarray(Image.open(str(image)).convert('L'))
            A.append(im)

    return A


def average_face_vector(face_vectors):
    """
    Given an array of image faces returns the average

    Keyword arguments:
    face_vectors -- array of images
    """

    dim = np.shape(face_vectors)
    average = np.zeros((1, dim[1]))
    for face_vector in face_vectors:
        average += face_vector
    average = average / dim[0]
    return average


def calculate_face_vectors(images):
    """
    Given an array of images return an array of vectors

    Keyword arguments:
    images -- array of images
    """

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
    """
    Get the training images
    """

    images = np.empty((args.subjects, args.img_per_subject, PIXELS_H * PIXELS_V))
    for i in range(0, args.subjects):
        training_faces = get_training_faces_for_subject(args, "s%d" % (i + 1))

        ''' Each face_vector is (1, 112*92) = (1, 10304)'''
        images[i] = calculate_face_vectors(training_faces)

    return images.reshape((args.subjects * args.img_per_subject, PIXELS_H * PIXELS_V))


def get_test_images(args):
    test_images = np.empty((args.subjects, args.test_img_per_subject, PIXELS_H * PIXELS_V))
    for i in range(0, args.subjects):
        test_faces = get_test_faces_for_subject(args, "s%d" % (i + 1))

        ''' Each face_vector is (1, 112*92) = (1, 10304)'''
        test_face_vectors = calculate_face_vectors(test_faces)
        test_images[i] = test_face_vectors

    return test_images.reshape((args.subjects * args.test_img_per_subject, PIXELS_H * PIXELS_V))

def get_image(path):
    return np.asarray(Image.open(path).convert('L'))

def pca(args):
    images = get_training_images(args)

    ''' Substract the media from every training image '''
    average_face_vect = average_face_vector(images)
    face_vectors_minus_avg = np.empty((images.shape[0], PIXELS_H * PIXELS_V))
    for i in range(images.shape[0]):
        face_vectors_minus_avg[i] = images[i] - average_face_vect

    test_images = get_test_images(args)

    ''' Substract the media from every testing image '''
    test_face_vectors_minus_avg = np.empty((test_images.shape[0], PIXELS_H * PIXELS_V))
    for i in range(test_images.shape[0]):
        test_face_vectors_minus_avg[i] = test_images[i] - average_face_vect

    ''' Calculate eigenvalues and eigenvectors of transpose(A)*A or A*transpose(A) according dimensions '''
    if len(face_vectors_minus_avg) > face_vectors_minus_avg[0].shape[0]:
        ''' #Rows > #Columns '''
        Ss = np.matrix(face_vectors_minus_avg).H * np.matrix(face_vectors_minus_avg)
        eigenvalues, eigenvectors = calculate_eigenvalues(Ss)
    else:
        ''' #Columns > #Rows '''
        Ss = np.matrix(face_vectors_minus_avg) * np.matrix(face_vectors_minus_avg).H
        eigenvalues, eigenvectors = calculate_eigenvalues(Ss)
        eigenvectors = face_vectors_minus_avg.transpose() * eigenvectors
        for i in range(face_vectors_minus_avg.shape[0]):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    improy = np.dot(face_vectors_minus_avg, eigenvectors)
    imtstproy = np.dot(test_face_vectors_minus_avg, eigenvectors)

    query_params = PCAQueryParams()
    query_params.average_image = average_face_vect
    query_params.eigenvectors = eigenvectors
    query_params.training_projections = improy
    query_params.test_projections = imtstproy

    return query_params


def kpca(args):
    images = get_training_images(args)
    # test_images = get_test_images(args)

    observations = args.subjects * args.img_per_subject

    degree = 2
    ''' Paper eq n° 61 '''
    K = (np.dot(images, images.T) + 1) ** degree

    ''' Center observations '''
    ''' Paper eq n° 49 '''
    ''' http://www.face-rec.org/algorithms/Kernel/kernelPCA_scholkopf.pdf '''
    ones = np.ones([observations, observations]) / observations
    K_dot_ones = np.dot(K, ones)
    K = K - np.dot(ones, K) - K_dot_ones + np.dot(ones, K_dot_ones)

    eigenvalues, eigenvectors = calculate_eigenvalues(np.matrix(K))

    ''' Eigenvalues come in descending order. We change that '''
    ''' Taken from Campus eg'''

    ''' Ordered eigenvalues '''
    eigenvectors = np.fliplr(eigenvectors)

    query_params = KernelPCAQueryParams()
    query_params.eigenvectors = eigenvectors

    ''' Paper eq n° 17 '''
    query_params.training_projections = np.dot(K.T, eigenvectors)

    query_params.images = images
    query_params.degree = degree
    query_params.ones = ones
    query_params.K = K
    query_params.observations = observations

    return query_params


def pca_query(args, image, clf, query_params: PCAQueryParams):
    image = np.array(matrix_to_vector(image))
    diff = image - query_params.average_image
    improy = np.dot([diff], query_params.eigenvectors)
    return clf.predict(improy)


def kpca_query(args, image, clf, query_params: KernelPCAQueryParams):
    image = np.array(matrix_to_vector(image))

    test_cases = 1
    ones_test = np.ones([test_cases, query_params.observations]) / query_params.observations

    ''' Paper eq n° 52 '''
    K_test = (np.dot([image], query_params.images.T) + 1) ** query_params.degree

    ''' Paper eq n° 54 '''
    ones_test_dot_K = np.dot(ones_test, query_params.K)
    K_test = K_test - ones_test_dot_K - np.dot(K_test, query_params.ones) + np.dot(ones_test_dot_K, query_params.ones)

    ''' Paper eq n° 51 '''
    imtstproypre = np.dot(K_test, query_params.eigenvectors)

    return clf.predict(imtstproypre)
