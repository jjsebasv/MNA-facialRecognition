#!/usr/bin/python3

from PIL import Image
import numpy as np
from pathlib import Path

def get_training_faces():
    trainingFaces = []
    imagesList = Path("images").glob('**/*.pgm')

    for image in imagesList:
        imageMatrix = np.asarray(Image.open(str(image)).convert('L'))
        imageMatrixReshaped = imageMatrix.reshape(-1, 1);
        trainingFaces.append(imageMatrixReshaped);
        print(imageMatrixReshaped.shape)

def main():
    get_training_faces()

if __name__ == "__main__":
    main()
