import os
import sys
import matplotlib.pyplot as plt
from subprocess import call
import numpy as np
import skimage.io
import PIL

import entity
import utils

counter = 0

class LMRContentOwner(entity.ContentOwner):

    MSBS = np.array([2,3,4,5,6,7,8])

    def __init__(self):
        super(LMRContentOwner, self).__init__()
        
    def encode_image(self, img: np.ndarray, secret_key: np.ndarray) -> dict:
        
        img = img.copy()
        
        h, w = img.shape
        
        # Construct the best location map
        img, msb_map_size, lm_size, msb, der = self.generate_maps(img, secret_key, self.MSBS).values()
        
        if msb:
            # encrypt the rotated image based on the generated secret key
            img = np.bitwise_xor(img, secret_key)
            encrypt_img = img.copy()
            
            compressed_lm = open('assets/temp/lm_map.jbg', 'rb').read()
            compressed_lm_bits = bin(int.from_bytes(compressed_lm, byteorder=sys.byteorder))[2:]
            
            compressed_msb_map = open('assets/temp/msb_map.jbg', 'rb').read()
            compressed_msb_map_bits = bin(int.from_bytes(compressed_msb_map, byteorder=sys.byteorder))[2:]
            
            # Concatenate the location map and msb_map together
            concatenated_maps = np.array(list(compressed_lm_bits + compressed_msb_map_bits), dtype=np.int)
            
            # Insert the concatenated maps into the most significant plane of the encrypted image
            img = img.flatten()
            img[:concatenated_maps.shape[0]] = np.where(
                concatenated_maps == 1, 
                img[:concatenated_maps.shape[0]] | 128, 
                img[:concatenated_maps.shape[0]] & 127
            )
            
            # Embed the location map size and msb map size into the last 36 pixels of the flattened encrypted image
            lm_size_bits = np.array(list(bin(lm_size)[2:].zfill(18)), dtype=np.int)
            msb_map_size_bits = np.array(list(bin(msb_map_size)[2:].zfill(18)), dtype = np.int)
           
            # Clear the most siginificant bit
            img[-36:]&= 127 
            
            # Embed the location map size information
            img[-18:] = np.where(
                lm_size_bits == 1, 
                img[-18:] | 128, 
                img[-18:] & 127
            )
            
            # Embed the msb map size information
            img[-36:-18] = np.where(
                msb_map_size_bits == 1, 
                img[-36:-18] | 128, 
                img[-36:-18] & 127
            )
            
            # Restore the original shape of the image after it was flattened
            img = img.reshape((h,w))
            
            return {'encrypted_img': img, 'before_embedding': encrypt_img, 'msb': msb, 'DER': der}
            
        else:
            return {}
        
        
class LMRDataHider(entity.DataHider):
    
    def __init__(self):
        super(LMRDataHider, self).__init__()
        
    def hiding_data(self, img: np.ndarray, msb: int) -> dict:
        
        img = img.copy()
        
        # extract the location map size
        lm_size = int("".join(np.array((img.flatten()[-18:] & 0x80) % 127, dtype=str)), 2)
                           
        # Extract the most significant bit plane
        msb_plane = ((img & 0x80) % 127).flatten()
        msb_plane_bits = "".join(np.array(msb_plane, dtype=str))
        lm_bytes = int(msb_plane_bits[:lm_size], 2).to_bytes((lm_size + 7) // 8, byteorder=sys.byteorder)
        
        # Write the location map JBG file for decompressing the location map
        with open('assets/temp/lm_map_hide_data.jbg', 'wb') as f:
            f.write(lm_bytes)
           
        call(['tools/jbgtopbm', 'assets/temp/lm_map_hide_data.jbg', 'assets/temp/lm_map_hide_data.pbm']) 
        
        # Decompress the location map
        lm = np.invert(np.array(PIL.Image.open('assets/temp/lm_map_hide_data.pbm'))).astype('uint8')
        
        # hide the secret information into the encrypted image
        np.random.seed(1)
        
        # Creating the offset given the msb
        offset = msb - 1
        
        # Create template given the offset
        template = int('1' + '0' * offset + '1' * (7 - offset), 2)
        
        # Creating the info based on the offset
        info = np.random.randint(0, (1 << offset), size=np.sum(lm==0)) # Generate random generated information bits
        
        shifted_info = info << (7 - offset) #  Shift the information bits to preserve the rest bits' values
        
        secret_key_2 = utils.crypto_tools.generate_secret_key_2(len(shifted_info), msb, emr=False)
        shifted_info = np.bitwise_xor(shifted_info, secret_key_2)
        
        # Applying the template and info when the location map == 0
        img[lm == 0] &= template # Clear the most siginificant bits, except the first most siginificant bits
        img[lm == 0] |= shifted_info # Do a bitwise OR operation to add the info bits
        
        return {'marked_encrypted_img': img, 'secret_key_2': secret_key_2, 'msb': msb, 'info': info}
class LMRRecipient(entity.Recipient):
    
    def __init__(self):
        super(LMRRecipient, self).__init__()
        
    def recover_image(self, img: np.ndarray, secret_key: np.ndarray, msb: int) -> np.ndarray:
        
        img = img.copy()
        
        h, w = img.shape

        limit = h if h > w else w # The limitation of block size

        # extract the location map size and the msb msp size
        lm_size = int("".join(np.array((img.flatten()[-18:] & 0x80) % 127, dtype=str)), 2)
        mm_size = int("".join(np.array((img.flatten()[-36:-18] & 0x80) % 127, dtype=str)), 2)
        
        # Extract the most significant bit plane
        msb_plane = ((img & 0x80) % 127).flatten()
        msb_plane_bits = "".join(np.array(msb_plane, dtype=str))
        
        lm_bytes = int(msb_plane_bits[:lm_size], 2).to_bytes((lm_size + 7) // 8, byteorder=sys.byteorder)
        mm_bytes = int(msb_plane_bits[lm_size:][:mm_size], 2).to_bytes((mm_size + 7) // 8,byteorder=sys.byteorder)
        
        # Write the location map and msb map JBG file for decompressing the location map
        with open('assets/temp/lm_map_recover_img.jbg', 'wb') as f:
            f.write(lm_bytes)
        with open('assets/temp/msb_map_recover_img.jbg', 'wb') as f:
            f.write(mm_bytes)
           
        # Decompress the location map and msb map
        call(['tools/jbgtopbm', 'assets/temp/lm_map_recover_img.jbg', 'assets/temp/lm_map_recover_img.pbm']) 
        call(['tools/jbgtopbm', 'assets/temp/msb_map_recover_img.jbg', 'assets/temp/msb_map_recover_img.pbm'])
        lm = np.invert(np.array(PIL.Image.open('assets/temp/lm_map_recover_img.pbm'))).astype('uint8')
        msb_map = np.invert(np.array(PIL.Image.open('assets/temp/msb_map_recover_img.pbm'))).astype('uint8')
        
        # Decrypt the image
        img ^= secret_key
        
        # Rotate the location map
        block_sizes = [2**(i+1) for i in range(3, limit) if 2**(i+1) <= limit][::-1] # block_sizes = [512, 256, 128, 64, 32, 16]

        # shuffle the location map based on the generated key
        for block_size in block_sizes:
            lm = utils.block_shuffle.block_shuffle(lm, secret_key, block_size, encrypt=False) # Rotate the location map
            msb_map = utils.block_shuffle.block_shuffle(msb_map, secret_key, block_size, encrypt=False) # Rotate the location map
            img = utils.block_shuffle.block_shuffle(img, secret_key, block_size, encrypt=False) # Rotate the image

        # Substitude the MSBs to origianl MSBs accroding to the msb_map
        img[msb_map == 1] |= 128
        img[msb_map == 0] &= 127
        
        # Create a 8-bit template of MSB 1s followed by MSB 0s
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
        
        # extract the location map size
        lm_size = int("".join(np.array((img.flatten()[-18:] & 0x80) % 127, dtype=str)), 2)
                           
        # Extract the most significant bit plane
        msb_plane = ((img & 0x80) % 127).flatten()
        msb_plane_bits = "".join(np.array(msb_plane, dtype=str))
        lm_bytes = int(msb_plane_bits[:lm_size], 2).to_bytes((lm_size + 7) // 8, byteorder=sys.byteorder)
        
        # Write the location map JBG file for decompressing the location map
        with open('assets/temp/lm_map_hide_data.jbg', 'wb') as f:
            f.write(lm_bytes)
           
        # Decompress the location map
        call(['tools/jbgtopbm', 'assets/temp/lm_map_hide_data.jbg', 'assets/temp/lm_map_hide_data.pbm']) 
        lm = np.invert(np.array(PIL.Image.open('assets/temp/lm_map_hide_data.pbm'))).astype('uint8')
        
        # hide the secret information into the encrypted image
        np.random.seed(1)
        
        # Creating the offset given the msb
        offset = msb - 1
        
        # Create template given the offset
        template = int('1' + '0' * offset + '1' * (7 - offset), 2)
        
        # Extract the information from the marked encrypted image
        info = ((img[lm == 0] | template) & 127) >> (7 - offset) << (7 - offset)
        
        info = np.bitwise_xor(info, secret_key) >> (7 - offset)
        
        return info
        
        
if __name__ == "__main__":

    img = skimage.io.imread('assets/images/lena.pgm')
    original_img = img.copy()
    
    secret_key = utils.crypto_tools.generate_secret_key_1(*img.shape)
    
    co = LMRContentOwner()
    encrypted_image, msb = co.encode_image(img, secret_key).values()
    dh = LMRDataHider()
    marked_encrypted_image, ms = dh.hiding_data(encrypted_image, msb).values()
    rp = LMRRecipient()
    rp.recover_image(marked_encrypted_image, secret_key, msb)
