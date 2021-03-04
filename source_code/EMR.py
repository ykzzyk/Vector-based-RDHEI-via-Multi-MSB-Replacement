import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import PIL

import entity
import utils
import random

class EMRContentOwner(entity.ContentOwner):

    MSBS = np.array([2,3,4,5,6,7])

    def __init__(self):
        super(EMRContentOwner, self).__init__()

    def encode_image(self, img: np.ndarray) -> dict:

        h, w = img.shape
        
        # Construct the best location map
        lm, msb = self.generate_location_map(img, self.MSBS)

        # Generate the secret key
        secret_key = utils.crypto_tools.generate_secret_key(*img.shape)
    
        # shuffle the location map based on the generated key
        block_sizes = [2,4,16,32,64,128,256,512]
        for block_size in block_sizes:
            lm = utils.block_shuffle.block_shuffle(lm, secret_key, block_size) # Rotate the location map
            img = utils.block_shuffle.block_shuffle(img, secret_key, block_size) # Rotate the image

        # encrypt the image based on the generated secret key
        img = np.bitwise_xor(img, secret_key)

        # Embed the location map into the encrypted image
        img[lm == 1] = np.bitwise_or(lm, img)[lm == 1]
        img[lm == 0] = np.bitwise_and((lm == 0) * 1 * 0xfe, img)[lm == 0]


        return {'encrypted_img': img, 'secret_key': secret_key, 'msb': msb}
        

        
class EMRDataHider(entity.DataHider):
    
    def __init__(self):
        super(EMRDataHider, self).__init__()
        
    def hiding_data(self, img: np.ndarray, msb: int) -> dict:
        
        # extract the location map
        lm = np.bitwise_and(1, img)

        # hide the secret information into the encrypted image
        np.random.seed(1)
        img[lm == 0] &= (1 << (8 - msb)) - 1
        info = np.random.randint(0, (1 << msb) - 1, size=np.sum(lm == 0)) << (8 - msb)
        img[lm == 0] |= info

        return {'marked_encrypted_img': img, 'msb': msb}
        
    
class EMRRecipient(entity.Recipient):
    
    def __init__(self):
        super(EMRRecipient, self).__init__()
        
    def recover_image(self, img: np.ndarray, secret_key: np.ndarray, msb: int) -> np.ndarray:

        h, w = img.shape

        # Extract the location map
        lm = np.bitwise_and(1, img)

        # Decrypt the image
        img ^= secret_key

        # Rotate the location map
        block_sizes = [512, 256, 128, 64, 32, 16, 4, 2]

        # shuffle the location map based on the generated key
        for block_size in block_sizes:
            lm = utils.block_shuffle.block_shuffle(lm, secret_key, block_size, encrypt=False) # Rotate the location map
            img = utils.block_shuffle.block_shuffle(img, secret_key, block_size, encrypt=False) # Rotate the image

        # template
        template = ((1 << msb) - 1) << (8 - msb)

        # Initilize the mark
        mark = np.zeros(shape=(h, ), dtype='uint8')

        # Image processing
        for i in range(w):

            mark[lm[:, i] == 1] = img[lm[:, i] == 1, i] & template
            img[lm[:, i] == 0, i] &= ((1 << (8 - msb)) - 1)
            img[lm[:, i] == 0, i] |= mark[lm[:, i] == 0]

        return img

    def extract_message(self, img: np.ndarray, msb: int) -> np.ndarray:

        # Extract the location map
        lm = np.bitwise_and(1, img)

        template = ((1 << msb) - 1) << (8 - msb)

        np.random.seed(1)

        # Extract the information from the marked image
        info = ((img[lm == 0] & template) >> (8 - msb)) << (8 - msb)

        return info

if __name__ == '__main__':

    # Initializing all the participants
    co = EMRContentOwner()
    dh = EMRDataHider()
    rp = EMRRecipient()

    # Load test image and create random encryption key
    img = skimage.io.imread(os.path.dirname(os.path.abspath(__file__)) + '/assets/images/lena.pgm')
    
    # Construct the RRBE scheme
    encoded_img, secret_key, msb = co.encode_image(img).values()
    marked_encoded_img, msb = dh.hiding_data(encoded_img, msb).values()
    message = rp.extract_message(marked_encoded_img, msb)
    recovered_img = rp.recover_image(marked_encoded_img, secret_key, msb)

    # Show the recovered image
    plt.imshow(recovered_img)
    plt.show()


