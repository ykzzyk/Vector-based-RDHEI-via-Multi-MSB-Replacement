import numpy as np
import pprint

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

A = np.linspace(0,24,25).reshape([5,5,])
B = get_blocks(A, 2)

pprint.pprint(B)