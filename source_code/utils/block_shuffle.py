from PIL import Image
import numpy as np
import skimage
import skimage.io
import math
import os
import sys
import time
import pathlib
import matplotlib.pyplot as plt

import crypto_tools as ct

def get_blocks(img, block_size):
    """Split a matrix into sub-matrices."""
    # Get original img's size
    h, w = img.shape

    # Trim elements not fit within the blocks
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)
    trim_array = img[:h_trim, :w_trim]

    # Reshape to stack the blocks and return these blocks
    return (trim_array.reshape(h//block_size, block_size, -1, block_size)
                      .swapaxes(1, 2)
                      .reshape(-1, block_size, block_size))

def combine_blocks(img, blocks):

    # Creating data container for reassembled img
    h, w = img.shape

    # Getting block quantity data
    nb, block_size, _ = blocks.shape
    
    n_h_b = int(np.floor(h/block_size)) # number of height blocks
    n_w_b = int(np.floor(w/block_size)) # number of width blocks
    
    trim_h = n_h_b * block_size
    trim_w = n_w_b * block_size

    # Reshape the blocks to reconstruct the image
    row_blocks = blocks.reshape((n_h_b, n_w_b, block_size, block_size))
    blocked_image = np.concatenate(np.concatenate(np.array_split(np.concatenate(row_blocks.swapaxes(0,1), axis=2), n_h_b)))
    
    # Account for non-blocked edges
    img[:trim_h, :trim_w] = blocked_image

    return img

def block_shuffle(img, sk, block_size, encrypt=True):

    # Obtain image information
    h, w = img.shape

    # Separate img and secret key into blocks
    img_blocks = get_blocks(img, block_size)
    sk_blocks = get_blocks(sk, block_size)

    # Perform reduction operation on secret key 
    sk_reduction = np.sum(sk_blocks, axis=(1,2)) % 4

    # Rotate img blocks if sk_reduction pair is 1, otherwise leave alone
    if encrypt:
        rot_map = {1:1, 2:2, 3:3}
    else:
        rot_map = {1:3, 2:2, 3:1}

    img_blocks[sk_reduction == 1] = np.rot90(img_blocks[sk_reduction == 1], k=rot_map[1], axes=(1,2))
    img_blocks[sk_reduction == 2] = np.rot90(img_blocks[sk_reduction == 2], k=rot_map[2], axes=(1,2))
    img_blocks[sk_reduction == 3] = np.rot90(img_blocks[sk_reduction == 3], k=rot_map[3], axes=(1,2))

    # Combine the blocks of the image and fill in any missing data from the original
    img = combine_blocks(img, img_blocks)

    return img