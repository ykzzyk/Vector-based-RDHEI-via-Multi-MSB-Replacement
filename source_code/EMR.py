import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import PIL

import entity
import utils

class EMRContentOwner(entity.ContentOwner):

    MSBS = np.array([2,3,4,5,6,7])

    def __init__(self):
        super(EMRContentOwner, self).__init__()

    def encode_image(self, img: np.ndarray, secret_key: np.ndarray) -> dict:

        h, w = img.shape
        
        # Construct the best location map
        lm, msb = self.generate_location_map(img, self.MSBS)
    
        # shuffle the location map based on the generated key
        block_sizes = [2,4,8,16,32,64,128,256,512]
        for block_size in block_sizes:
            lm = utils.block_shuffle.block_shuffle(lm, secret_key, block_size) # Rotate the location map
            img = utils.block_shuffle.block_shuffle(img, secret_key, block_size) # Rotate the image

        # encrypt the image based on the generated secret key
        img = np.bitwise_xor(img, secret_key)
        encrypt_img = img.copy()

        # Embed the location map into the encrypted image
        img[lm == 1] = np.bitwise_or(lm, img)[lm == 1]
        img[lm == 0] = np.bitwise_and((lm == 0) * 1 * 0xfe, img)[lm == 0]


        return {'encrypted_img': img, 'before_embedding': encrypt_img, 'msb': msb}
        

        
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
        
        
        
        
        
        
        origianl_info = info.copy()
        
        secret_key_2 = utils.crypto_tools.generate_secret_key_2(len(info), msb)
        info = np.bitwise_xor(info, secret_key_2)
        
        print(secret_key_2)
        print(info)
        img[lm == 0] |= info

        return {'marked_encrypted_img': img, 'secret_key_2': secret_key_2, 'msb': msb, 'original_info': origianl_info}
        
    
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
        block_sizes = [512, 256, 128, 64, 32, 16, 8, 4, 2]

        # shuffle the location map and marked decrypted image based on the generated key
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

    def extract_message(self, img: np.ndarray, secret_key: np.ndarray, msb: int) -> np.ndarray:

        # Extract the location map
        lm = np.bitwise_and(1, img)

        template = ((1 << msb) - 1) << (8 - msb)

        np.random.seed(1)

        # Extract the information from the marked encrypted image
        info = ((img[lm == 0] & template) >> (8 - msb)) << (8 - msb)
        
        info = np.bitwise_xor(info, secret_key)

        return info

# Automation 100000 tests
if __name__ == '__main__':
    pass
    