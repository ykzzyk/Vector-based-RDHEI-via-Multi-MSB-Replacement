from subprocess import call
from PIL import Image  # Import Image from Pillow mudule
import numpy as np  # Import Numpy
import pickle  # Store the key1
import os  # Generate cryptographically secure RNG: os.urandom()
import sys  # Import sys
import random

def generate_secret_key_1(img_h: int, img_w: int) -> np.ndarray:

    # Determining the size of the image
    key_size = img_h * img_w

    # Creating cryptographically-safe random number generator
    cryptogen = random.SystemRandom()

    # Generate 1D secret key
    secret_key = np.array([cryptogen.randrange(0,256) for _ in range(key_size)])

    # Reshape the 1D secret key to match the image
    secret_key = secret_key.reshape((img_h, img_w))
    
    return secret_key

def generate_secret_key_2(key_size: int, msb: int, emr=True) -> np.ndarray:
    
    # Creating cryptographically-safe random number generator
    cryptogen = random.SystemRandom()

    # Generate 1D secret key
    if emr:
        secret_key = np.array([cryptogen.randrange(0,(1 << msb) - 1) for _ in range(key_size)]) << (8 - msb)
    else:
        offset = msb - 1
        secret_key = np.array([cryptogen.randrange(0,(1 << offset)) for _ in range(key_size)]) << (7 - offset)
    return secret_key