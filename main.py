#!/usr/bin/python3
import argparse

from sklearn import svm

from face_recognition import *

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

    return parser.parse_args()


def main():
    args = parse_args()

    # clf, avg_image, V = pca(args)

    query_params = METHODS[args.method][TRAINING_FUNCTION](args)

    person = np.array([[i + 1] * args.img_per_subject for i in range(args.subjects)])
    # persontst = np.array([[i + 1] * args.test_img_per_subject for i in range(args.subjects)])

    clf = svm.LinearSVC()
    clf.fit(query_params.training_projections, person.ravel())

    acc = 0
    total = 0
    for subject in range(1, args.subjects + 1):
        for image in range(0, args.test_img_per_subject):
            total += 1
            ans = METHODS[args.method][TEST_FUNCTION](args, subject, image, clf, query_params)
            print(ans, subject)
            if ans == subject:
                acc += 1

    print(float(acc) / total)


if __name__ == "__main__":
    main()
