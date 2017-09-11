from pathlib import Path
from utils import *


def get_training_faces():
    c = 0
    A, y = [], []
    images_list = Path("images/s1").glob('**/*.pgm')

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
