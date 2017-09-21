#!/usr/bin/python3
import argparse
import pprint

from sklearn import svm

from face_recognition import *
from serialization_utils import *

TRAINING_FUNCTION = 0
TEST_FUNCTION = 1

METHODS = {
    'pca': (pca, pca_query),
    'kpca': (kpca, kpca_query)
}

IMAGE_DATABASES = [
    'att_images',
    'images'
]

SUBJECTS = 2
IMG_PER_SUBJECT = 6
TEST_IMG_PER_SUBJECT = 4


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--method',
        help='Method used to perform the face recognition',
        default="pca",
        choices=METHODS.keys()
    )

    parser.add_argument(
        '--imgdb',
        help='Image database',
        default='att_images',
        choices=IMAGE_DATABASES
    )

    parser.add_argument(
        '--subjects',
        help="Number of subjects",
        default=SUBJECTS,
        type=int
    )

    parser.add_argument(
        '--img-per-subj',
        help="Number of training images per subject",
        default=IMG_PER_SUBJECT,
        dest='img_per_subject',
        type=int
    )

    parser.add_argument(
        '--test-img-per-subj',
        help="Number of training images per subject",
        default=TEST_IMG_PER_SUBJECT,
        dest='test_img_per_subject',
        type=int
    )

    parser.add_argument(
        '--query',
        help="Path of file to query"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # clf, avg_image, V = pca(args)

    if check_query_params(args):
        query_params = load_query_params(args)
    else:
        query_params = METHODS[args.method][TRAINING_FUNCTION](args)
        save_query_params(query_params, args)

    person = np.array([[i + 1] * args.img_per_subject for i in range(args.subjects)])
    # persontst = np.array([[i + 1] * args.test_img_per_subject for i in range(args.subjects)])

    clf = svm.LinearSVC()
    clf.fit(query_params.training_projections, person.ravel())

    if args.query is None:
        acc = 0
        total = 0
        for subject in range(1, args.subjects + 1):
            for image in range(0, args.test_img_per_subject):
                total += 1
                image = get_test_faces_for_subject(args, "s%s" % subject)[image]
                ans = METHODS[args.method][TEST_FUNCTION](args, image, clf, query_params)
                print(ans, subject)
                if ans == subject:
                    acc += 1

        print(float(acc) / total)
    else:
        image = get_image(args.query)
        ans = METHODS[args.method][TEST_FUNCTION](args, image, clf, query_params)
        print("Sujeto: " + str(ans))


if __name__ == "__main__":
    main()
