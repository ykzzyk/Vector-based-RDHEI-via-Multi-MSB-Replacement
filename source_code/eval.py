import argparse
import skimage.io
import matplotlib.pyplot as plt
from skimage import metrics
import numpy as np
import math
from scipy.stats import chisquare
import warnings

import EMR
import LMR
import utils

def shannon_entropy(img, h, w):
    _, counts = np.unique(img, return_counts=True)
    p = (counts / (h * w))
    p *= np.log2(p)
    return (-np.sum(p))
    
def chi_square(img, h, w):
    _, counts = np.unique(img, return_counts=True)
    
    prob = (counts / (h * w)) - 1/256
    temp = np.sum(np.power(prob, 2))
    
    x_2 = np.sqrt(256 * h * w * temp)
    
    return x_2
    
def npcr(img1, img2, h, w):
    res = (np.count_nonzero(np.equal(img1, img2) == 0)) / (h * w)
    return res
    
    
def uaci(img1, img2, h, w):
    res = np.sum(np.abs(img1 - img2) / 255) / (h * w)
    return res

def parser_arguments():
    parser = argparse.ArgumentParser(prog='RDHEI-EMR-LMR',
                                     usage="%(prog)s [-h] [method: type 'EMR' or 'LMR']",
                                     description='Reversible data hiding in encrypted images')

    parser.add_argument('method',
                        metavar='method',
                        type=str,
                        help="type 'EMR' or 'LMR'")
    parser.add_argument('name', 
                        metavar='name', 
                        type=str, 
                        help='The test image name')

    args = parser.parse_args()
    
    return args.method.upper(), args.name


if __name__ == '__main__':

    method, name = parser_arguments()
    
    print(f"----- Test the image {name} -----\n")

    image_path = f'assets/images/{name}.pgm'
    img = skimage.io.imread(image_path)
    h, w = img.shape

    if method == 'EMR':
        content_owner = EMR.EMRContentOwner()
        data_hider = EMR.EMRDataHider()
        recipient = EMR.EMRRecipient()
    elif method == 'LMR':
        content_owner = LMR.LMRContentOwner()
        data_hider = LMR.LMRDataHider()
        recipient = LMR.LMRRecipient()
    
    # Perform the corresponding method based on the user input
    # Construct the RRBE scheme
    # Generate the secret key
    secret_key = utils.crypto_tools.generate_secret_key_1(*img.shape)

    encoded_img, encrypt_img, msb = content_owner.encode_image(img, secret_key).values()
    
    marked_encoded_img, secret_key_2, msb, info = data_hider.hiding_data(encoded_img, msb).values()
      
    message = recipient.extract_message(marked_encoded_img, secret_key_2, msb)
    print(f"\n----- Secret Information Extraction Phase -----\nIs it error-free? Answer: {(info == message).all()}")
    
    recovered_img = recipient.recover_image(marked_encoded_img, secret_key, msb)
    
    print("\n----- Shannon Entropy Results -----")
    
    # Calculate the Shannon Entropy
    se = shannon_entropy(img, h, w)
    print(f"The Shannon Entropy of the original image is: {se}")
    se = shannon_entropy(encrypt_img, h, w)
    print(f"The Shannon Entropy of the encrypted image is: {se}")
    se = shannon_entropy(marked_encoded_img, h, w)
    print(f"The Shannon Entropy of the marked encrypted image is: {se}")
    
    print("\n----- Chi Square Results -----")
    
    # Calculate the Chi Square
    cs = chi_square(img, h, w)
    print(f"The chisquare of the original image is: {cs}") 
    cs = chi_square(encrypt_img, h, w)
    print(f"The chisquare of the encrypted image is: {cs}")
    cs = chi_square(marked_encoded_img, h, w)
    print(f"The chisquare of the marked encrypted image is: {cs}")
    
    print("\n----- NPCR Results -----")
    
    # Calculate the NPCR
    n = npcr(img, encrypt_img, h, w)
    print(f'The NPCR between original image and encrypted image is: {n}')
    n = npcr(img, marked_encoded_img, h, w)
    print(f'The NPCR between original image and marked encrypted image is: {n}')
    
    print("\n----- UACI Results -----")
    
    # Calculate the UACI
    u = uaci(img, encrypt_img, h, w)
    print(f'The UACI between original image and encrypted image is: {u}')
    u = uaci(img, marked_encoded_img, h, w)
    print(f'The UACI between original image and marked encrypted image is: {u}')
    
    print("\n----- PSNR Results -----")
    
    # Calculate the PSNR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psnr = metrics.peak_signal_noise_ratio(img, encrypt_img, data_range=None)
        print(f"The Peak Signal-to-Noise Ratio between original image and encrypted image is: {psnr}")
        psnr = metrics.peak_signal_noise_ratio(img, marked_encoded_img, data_range=None)
        print(f"The Peak Signal-to-Noise Ratio between original image and marked encrypted image is: {psnr}")
        psnr = metrics.peak_signal_noise_ratio(img, recovered_img, data_range=None)
        print(f"The Peak Signal-to-Noise Ratio between original image and recovered image is: {psnr}")
        
    print("\n----- SSIM Results -----")
    
    # Calculate the SSIM
    ssim = metrics.structural_similarity(img, recovered_img, data_range=recovered_img)
    print(f"The Structural SIMilarity between original image and recovered image is: {ssim}\n")
    