#!/usr/bin/python3
import argparse
import pprint

from sklearn import svm

from face_recognition import *
from serialization_utils import *

WEBCAM_AVAILABLE = False
try:
    import pygame
    import pygame.camera
    import tempfile

    WEBCAM_AVAILABLE = True
except ImportError:
    print("Webcam libraries not available. Install pygame")

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

SUBJECTS = 5
IMG_PER_SUBJECT = 9
TEST_IMG_PER_SUBJECT = 1


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

    if WEBCAM_AVAILABLE:
        parser.add_argument(
            '-l',
            '--live',
            help="Live subject recognition",
            action='store_true'
        )

        parser.add_argument(
            '--webcams',
            help="Query video devices",
            action='store_true'
        )
        parser.add_argument(
            '--webcam',
            help="Video device",
            default='/dev/video0'
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

        if WEBCAM_AVAILABLE and args.webcams:
            pygame.camera.init()
            for camera in pygame.camera.list_cameras():
                print("Device found: %s" % camera)
            exit(0)

        if WEBCAM_AVAILABLE and args.live:
            # Query the db using the webcam
            from read_image import DEFAULT_IMAGE_SIZE, resize_and_crop
            pygame.camera.init()
            cam = pygame.camera.Camera(args.webcam, )

            try:
                while True:
                    input("Press enter to capture an image")
                    print("Capturing...")
                    cam.start()
                    img = cam.get_image()
                    cam.stop()

                    file, path = tempfile.mkstemp()
                    pygame.image.save(img, path)
                    resize_and_crop(path, DEFAULT_IMAGE_SIZE, 'middle')

                    image = get_image(path + ".pgm")
                    ans = METHODS[args.method][TEST_FUNCTION](args, image, clf, query_params)
                    if args.imgdb == 'images':
                        print("Sujeto: " + subject_name(ans[0]))
                    else:
                        print("Sujeto: " + str(ans))

            except KeyboardInterrupt:
                print("\nExiting")
                #cam.stop()
                exit(0)

        else:

            # Query the db using the remaining images
            acc = 0
            total = 0
            for subject in range(1, args.subjects + 1):
                for image in range(0, args.test_img_per_subject):
                    total += 1
                    image = get_test_faces_for_subject(args, "s%s" % subject)[image]
                    ans = METHODS[args.method][TEST_FUNCTION](args, image, clf, query_params)
                    if args.imgdb == 'images':
                        print(subject_name(ans[0]), "\tshould've been\t", subject_name(subject))
                    else:
                        print(str(ans), "\tshould've been\t", str(subject))

                    if ans == subject:
                        acc += 1

            print(float(acc) / total)
    else:
        # Lets query using an image file
        image = get_image(args.query)
        ans = METHODS[args.method][TEST_FUNCTION](args, image, clf, query_params)
        if args.imgdb == 'images':
            print("Sujeto: " + subject_name(ans[0]))
        else:
            print("Sujeto: " + str(ans))


if __name__ == "__main__":
    main()
