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

# Local imports 
import visualize as vz
import crypto_tools as ct

#-------------------------------------------------------------------------------
# Old Block Shuffling Function

# def rot(arr, n, x1, y1): #this is the function which rotates a given block
#     temple = []
#     for i in range(n):
#         temple.append([])
#         for j in range(n):
#             temple[i].append(arr[x1+i, y1+j])
#     for i in range(n):
#         for j in range(n):
#             arr[x1+i,y1+j] = temple[n-1-i][n-1-j]


# def old_block_shuffle(img, block_size=5):

#     new_img = img.copy()
#     h, w = new_img.shape

#     for i in range(2, block_size+1):
#         for j in range(int(math.floor(float(h)/float(i)))):
#             for k in range(int(math.floor(float(w)/float(i)))):
#                 rot(new_img, i, j*i, k*i)

#     for i in range(3, block_size+1):
#         for j in range(int(math.floor(float(h)/float(block_size+2-i)))):
#             for k in range(int(math.floor(float(w)/float(block_size+2-i)))):
#                 rot(new_img, block_size+2-i, j*(block_size+2-i), k*(block_size+2-i))

#     return new_img

#-------------------------------------------------------------------------------
# New simple Block Shuffling Routines

def get_blocks(array, block_size):
    """Split a matrix into sub-matrices."""
    # Get original array's size
    h, w = array.shape

    # Trim elements not fit within the blocks
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)
    trim_array = array[:h_trim, :w_trim]

    # Reshape to stack the blocks and return these blocks
    return (trim_array.reshape(h//block_size, block_size, -1, block_size)
                      .swapaxes(1, 2)
                      .reshape(-1, block_size, block_size))

def combine_blocks(blocks, img):

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
    #print(sk_reduction)

    # Rotate img blocks if sk_reduction pair is 1, otherwise leave alone
    if encrypt:
        rot_map = {1:1, 2:2, 3:3}
    else:
        rot_map = {1:3, 2:2, 3:1}

    img_blocks[sk_reduction == 1] = np.rot90(img_blocks[sk_reduction == 1], k=rot_map[1], axes=(1,2))
    img_blocks[sk_reduction == 2] = np.rot90(img_blocks[sk_reduction == 2], k=rot_map[2], axes=(1,2))
    img_blocks[sk_reduction == 3] = np.rot90(img_blocks[sk_reduction == 3], k=rot_map[3], axes=(1,2))

    # Combine the blocks of the image and fill in any missing data from the original
    # image
    img = combine_blocks(img_blocks, img)

    return img

#-------------------------------------------------------------------------------
if __name__ == '__main__':

    # Loading image
    img_path = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent / 'assets' / 'location_maps' / 'lm_lena.pgm'
    img = np.array(Image.open(str(img_path)))

    # Setting blocksize
    #block_sizes = [2,4,16,32,64,128,256,128,64,32,16,4,2]
    #block_sizes = [2,4,16,32,64,128,256,512]
    block_sizes = [2]

    # ! Debugging purposes
    #img = img[:6, :6]

    # Create secret_key 
    tic = time.time()
    sk = ct.generate_secret_key(*img.shape)
    toc = time.time()
    diff = tic-toc

    # Block Shuffling the image
    shuffling_imgs = []
    new_img = img.copy()
    for block_size in block_sizes:
        new_img = block_shuffle(new_img, sk, block_size, encrypt=True)
        shuffling_imgs.append(new_img.copy())

    shuffling_img = shuffling_imgs[-1]

    # Unshuffle the image
    unshuffling_imgs = []
    unshuffle_img = new_img.copy()
    for block_size in block_sizes[::-1]:
        unshuffle_img = block_shuffle(unshuffle_img, sk, block_size, encrypt=False)
        unshuffling_imgs.append(unshuffle_img)

    restored_img = unshuffling_imgs[-1]
    print((img == restored_img).all())

    # Visualize image
    vz.make_summary_figure(
        b=np.expand_dims(shuffling_img, axis=0),
        c=np.expand_dims(restored_img, axis=0),
        a=np.expand_dims(img, axis=0))
    plt.show()
    
