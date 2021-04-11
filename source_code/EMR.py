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
        
        # original_img = img.copy()
        img = img.copy()

        h, w = img.shape
        
        # Construct the best location map
        lm, msb, bpp = self.generate_location_map(img, self.MSBS)
        # lm_before_rotating = lm.copy()
    
        # shuffle the location map based on the generated key
        block_sizes = [2,4,8,16,32,64,128,256,512]
        for block_size in block_sizes:
            lm = utils.block_shuffle.block_shuffle(lm, secret_key, block_size) # Rotate the location map
            img = utils.block_shuffle.block_shuffle(img, secret_key, block_size) # Rotate the image
            
        # # Plot the images
        # fig = plt.figure(1, figsize=(10,8))
        # axis1 = plt.subplot(221)
        # plt.imshow(original_img)
        # axis1.set_title('Original Image Before Rotating')

        # axis2 = plt.subplot(222)
        # plt.imshow(lm_before_rotating)
        # axis2.set_title('Location Map Before Rotating')
        
        # axis3 = plt.subplot(223)
        # plt.imshow(img)
        # axis3.set_title('Original Image After Rotating')
        
        # axis4 = plt.subplot(224)
        # plt.imshow(lm)
        # axis4.set_title('Location Map After Rotating')
        
        # plt.show()

        # encrypt the image based on the generated secret key
        img = np.bitwise_xor(img, secret_key)
        encrypt_img = img.copy()

        # Embed the location map into the encrypted image
        img[lm == 1] = np.bitwise_or(lm, img)[lm == 1]
        img[lm == 0] = np.bitwise_and((lm == 0) * 1 * 0xfe, img)[lm == 0]


        return {'encrypted_img': img, 'before_embedding': encrypt_img, 'msb': msb, 'DER': bpp}
        

        
class EMRDataHider(entity.DataHider):
    
    def __init__(self):
        super(EMRDataHider, self).__init__()
        
    def hiding_data(self, img: np.ndarray, msb: int) -> dict:
        
        img.copy()
        
        # extract the location map
        lm = np.bitwise_and(1, img)

        # hide the secret information into the encrypted image
        np.random.seed(1)
        img[lm == 0] &= (1 << (8 - msb)) - 1 # Clear the b-MSB bits to hide the future message inside it
        
        info = np.random.randint(0, (1 << msb) - 1, size=np.sum(lm == 0)) # Generate random information
        
        shifted_info = info << (8 - msb) # Shift the information bits to preserve the rest bits' values
        
        secret_key_2 = utils.crypto_tools.generate_secret_key_2(len(shifted_info), msb)
        shifted_info = np.bitwise_xor(shifted_info, secret_key_2)
  
        img[lm == 0] |= shifted_info

        return {'marked_encrypted_img': img, 'secret_key_2': secret_key_2, 'msb': msb, 'info': info}
        
    
class EMRRecipient(entity.Recipient):
    
    def __init__(self):
        super(EMRRecipient, self).__init__()
        
    def recover_image(self, img: np.ndarray, secret_key: np.ndarray, msb: int) -> np.ndarray:
        
        img = img.copy()

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
        
        img = img.copy()

        # Extract the location map
        lm = np.bitwise_and(1, img)

        template = ((1 << msb) - 1) << (8 - msb)

        np.random.seed(1)

        # Extract the information from the marked encrypted image
        info = (img[lm == 0] & template) >> (8 - msb) << (8 - msb)
        
        info = np.bitwise_xor(info, secret_key) >> (8 - msb)

        return info

    