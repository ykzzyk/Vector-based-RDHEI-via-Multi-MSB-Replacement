import os
import sys
import abc
from subprocess import call
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import PIL

import entity
import utils

# This code should reflect the global scheme of the EMR and LMR

class ContentOwner(abc.ABC):

    @abc.abstractmethod
    def encode_image(self, img: np.ndarray) -> dict:
        ...

    def generate_location_map(self, img: np.ndarray, msb: np.ndarray) -> (np.ndarray, int):
        """
        Generates all possible location maps and determines the best one based
        on its bpp.
        """

        h, w = img.shape
        msbs = msb.shape[0]

        # Expand image to account for multiple location maps
        e_img = np.expand_dims(img, axis=0)

        # Create a 8-bit template of MSB 1s followed by MSB 0s
        scalar_mask = ((1 << msb) - 1) << (8 - msb)
        mask = np.resize(scalar_mask,  (h, msbs)).T
        past_data = np.zeros_like(mask)

        # Create location map
        lms = np.zeros((msbs, h, w), dtype='uint8')

        # Set initial condition
        lms[..., 0] = 1
        past_data[..., :] = e_img[..., 0]

        # Image processing
        for i in range(1,w):

            msb_changed = (e_img[..., i] & mask != past_data & mask)
            past_data = np.where(msb_changed, e_img[...,i], past_data)
            lms[msb_changed, i] = 1

        # Select the best location map
        _, h, w = lms.shape

        # Calculate the bpp for all and select the best
        bpps = (msb / (h * w)) * np.sum(lms == 0, axis=(1,2))
        max_index = np.argmax(bpps)
        
        max_bpp = bpps[max_index]
        max_msb = msb[max_index]
        max_lm = lms[max_index]

        return max_lm, max_msb, max_bpp
    
    def generate_maps(self, img: np.ndarray, secret_key: np.ndarray, msb: np.ndarray) -> dict:
        
        """
        Generate the most significant map (MSB map)
        """
        
        # img_before_rotating = img.copy()
        
        # Create the most significant map (msb_map)
        # msb_map_before_rotating = (img & 0x80) % 127
        # msb_map = msb_map_before_rotating.copy()
        
        msb_map = (img & 0x80) % 127
        
        """
        Generate the most optimal location map and the MSB map
        """
        
        h, w = img.shape
        msbs = msb.shape[0]
        
        # Expand image to account for multiple location maps
        e_img = np.expand_dims(img, axis=0)

        # Create a 8-bit template of MSB 1s followed by MSB 0s
        scalar_mask = ((1 << msb) - 1) << (8 - msb)
        mask = np.resize(scalar_mask,  (h, msbs)).T
        past_data = np.zeros_like(mask)
        
        # Create location map
        lms = np.zeros((msbs, h, w), dtype='uint8')
        
        # Set initial condition
        lms[..., 0] = 1
        past_data[..., :] = e_img[..., 0]

        # Image processing
        for i in range(1,w):

            msb_changed = (e_img[..., i] & mask != past_data & mask)
            past_data = np.where(msb_changed, e_img[...,i], past_data)
            lms[msb_changed, i] = 1

        # Select the best location map
        _, h, w = lms.shape
        
        
        # Calculate the bpp for all and select the best
        bpps = ((msb - 1) / (h * w)) * np.sum(lms == 0, axis=(1,2))
        
        np.set_printoptions(threshold=sys.maxsize)
        
        while bpps.shape[0] != 0:
            max_index = np.argmax(bpps)

            max_bpp = bpps[max_index]
            max_msb = msb[max_index]
            max_lm = lms[max_index]
            # lm_before_rotating = max_lm.copy()
            
            block_sizes = [16, 32, 64, 128, 256, 512]
            for block_size in block_sizes:
                max_lm = utils.block_shuffle.block_shuffle(max_lm, secret_key, block_size) # Rotate the location map
                if bpps.shape[0] == 7:
                    msb_map = utils.block_shuffle.block_shuffle(msb_map, secret_key, block_size) # Rotate the msb map
                    rotated_img = utils.block_shuffle.block_shuffle(img, secret_key, block_size) # Rotate the original img
            # Employ JBIG-KIT to compress the two maps - MSB_map, Location_map
            msb_map_str = str(msb_map.flatten()).replace('[', ' ').replace(']', ' ')
            max_lm_str = str(max_lm.flatten()).replace('[', ' ').replace(']', ' ')
            with open(f'assets/temp/msb_map.pbm', 'w+') as f:
                f.write(f'P1\n512\n512\n\n{msb_map_str}')
                
            with open(f'assets/temp/lm_map.pbm', 'w+') as f:
                f.write(f'P1\n512\n512\n\n{max_lm_str}')

            call(['tools/pbmtojbg', '-q', 'assets/temp/msb_map.pbm', 'assets/temp/msb_map.jbg'])
            call(['tools/pbmtojbg', '-q', 'assets/temp/lm_map.pbm', 'assets/temp/lm_map.jbg'])
            
            # The size of the compressed msb map
            compressed_msb_map_size = os.stat('assets/temp/msb_map.jbg').st_size * 8 - 6
            
            # The size of the compressed location map size
            compressed_location_map_size = os.stat('assets/temp/lm_map.jbg').st_size * 8 - 6

            # Check the total size of the compressed msb_map and the compressed location map
            total_size = compressed_msb_map_size + compressed_location_map_size
            
            if total_size < (h * w - 36):
                
                # # Plot the figures
                # fig = plt.figure(1, figsize=(14,8))
                # axis1 = plt.subplot(231)
                # plt.imshow(lm_before_rotating)
                # axis1.set_title('The Optimal Location Map Before Rotating')
                # axis2 = plt.subplot(234)
                # plt.imshow(max_lm)
                # axis2.set_title('The Optimal Location Map After Rotating')

                # axis3 = plt.subplot(232)
                # plt.imshow(msb_map_before_rotating)
                # axis3.set_title('The First MSB Map Before Rotating')
                # axis4 = plt.subplot(235)
                # plt.imshow(msb_map)
                # axis4.set_title('The First MSB Map After Rotating')
                
                # axis5 = plt.subplot(233)
                # plt.imshow(img_before_rotating)
                # axis5.set_title('The Original image Before Rotating')
                # axis6 = plt.subplot(236)
                # plt.imshow(rotated_img)
                # axis6.set_title('The Original image After Rotating')
                
                # plt.show()
                
                return {"rotated_img": rotated_img, 
                        "compressed_msb_map_size": compressed_msb_map_size, 
                        "compressed_location_map_size": compressed_location_map_size, 
                        "max_msb": max_msb,
                        "max_bpp": max_bpp}
            else:
                bpps = np.delete(bpps, max_index)
                msb = np.delete(msb, max_index)
                lms = np.delete(lms, max_index, axis=0)
                    
        return {"rotated_img": None, 
                "compressed_msb_map_size": None, 
                "compressed_location_map_size": None, 
                "max_msb": None,
                "max_bpp": None}
        
class DataHider(abc.ABC):
    
    @abc.abstractmethod
    def hiding_data(self, img: np.ndarray, msb: int) -> np.ndarray:
        ...
    
class Recipient(abc.ABC):
    
    @abc.abstractmethod
    def recover_image(self, img: np.ndarray, secret_key: np.ndarray, msb: int) -> np.ndarray:
        ...
          
    @abc.abstractmethod
    def extract_message(self, img: np.ndarray, msb: int) -> np.ndarray:
        ...