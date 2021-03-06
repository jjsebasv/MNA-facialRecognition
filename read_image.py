# Code based on https://gist.github.com/sigilioso/2957026

from PIL import Image
from pathlib import Path

DEFAULT_IMAGE_SIZE = [92, 112]


def resize_and_crop(img_path, size, crop_type='top'):
    """
    Resize and crop an image to fit the specified size.
    args:
        img_path: path for the image to resize.
        modified_path: path to store the modified image.
        size: `(width, height)` tuple.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'midle' or
            'bottom/rigth' of the image to fit the size.
    raises:
        Exception: if can not open the file in img_path of there is problems
            to save the image.
        ValueError: if an invalid `crop_type` is provided.
    """

    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)

    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])

    # The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], int(size[0] * img.size[1] / img.size[0])),
                         Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, (int(img.size[1] - size[1]) / 2), img.size[0], int((img.size[1] + size[1]) / 2))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(size[1] * img.size[0] / img.size[1]), size[1]),
                         Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (int((img.size[0] - size[0]) / 2), 0, int((img.size[0] + size[0]) / 2), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]),
                         Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    img.convert('LA').convert('RGB')
    img.save(str(img_path) + ".pgm")


def get_images_from_files():
    """
    Resize and crop all the images from webcam_images
    """

    images_list = Path("webcam_images/").glob('**/*.jpg')

    for image in images_list:
        resize_and_crop(image, DEFAULT_IMAGE_SIZE, 'middle')

    print("Images ready!")


get_images_from_files()
