#!/usr/bin/python3

from PIL import Image
import numpy as np

def main():
    imageMatrix = np.asarray(Image.open('images/s1/1.pgm').convert('L'))
    print(imageMatrix )

if __name__ == "__main__":
    main()
