from subprocess import call
from PIL import Image  # Import Image from Pillow mudule
import numpy as np  # Import Numpy
import pickle  # Store the key1
import os  # Generate cryptographically secure RNG: os.urandom()
import sys  # Import sys
import random

def generate_secret_key(img_h: int, img_w: int) -> np.ndarray:

    # Determining the size of the image
    key_size = img_h * img_w

    # Creating cryptographically-safe random number generator
    cryptogen = random.SystemRandom()

    # Generate 1D secret key
    secret_key = np.array([cryptogen.randrange(0,256) for _ in range(key_size)])

    # Reshape the 1D secret key to match the image
    secret_key = secret_key.reshape((img_h, img_w))
    
    return secret_key