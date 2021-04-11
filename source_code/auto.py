import argparse
import skimage.io
import os
import matplotlib.pyplot as plt
import numpy as np
import math

import EMR
import LMR
import utils
from pathlib import Path

def parser_arguments():
    parser = argparse.ArgumentParser(prog='RDHEI-EMR-LMR',
                                     usage="%(prog)s [-h] [method: type 'EMR' or 'LMR']",
                                     description='Reversible data hiding in encrypted images')

    parser.add_argument('method',
                        metavar='method',
                        type=str,
                        help="type 'EMR' or 'LMR'")


    args = parser.parse_args()
    
    return args.method.upper()

if __name__ == "__main__":
    
    dir_emr = {
        "image_name": [],
        "DER": [],
        "MSB": [],
        "PSNR": [],
        "SSIM": []
    }
    
    dir_lmr = {
        "image_name": [],
        "DER": [],
        "MSB": []
    }
    
    for i in range(10001):
        image_path = Path.cwd() / 'assets' / 'BOWS2' / f'{i}.pgm'
        img = skimage.io.imread(image_path)
        h, w = img.shape

        if method == 'EMR':
            content_owner = EMR.EMRContentOwner()
            recipient = EMR.EMRRecipient()
        elif method == 'LMR':
            content_owner = LMR.LMRContentOwner()
            
        
        secret_key_1 = utils.crypto_tools.generate_secret_key_1(*img.shape)

        encoded_img, encrypt_img, msb = content_owner.encode_image(img, secret_key_1).values()
        
        dir_emr["image_name"],.append(f'{i}.pgm')
        dir_emr["image_name"],.append(f'{i}.pgm')
        
        recovered_img = recipient.recover_image(encoded_img, secret_key_1, msb)
        
        
        
    