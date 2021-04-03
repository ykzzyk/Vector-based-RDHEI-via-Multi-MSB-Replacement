import argparse
import skimage.io
import matplotlib.pyplot as plt
from skimage import metrics
import numpy as np
import math
from scipy.stats import chisquare

import EMR
import LMR
import utils

def shannon_entropy(img, h, w):
    _, counts = np.unique(img, return_counts=True)
    p = (counts / (h * w))
    p *= np.log2(p)
    print(f"The Shannon Entropy is: {-np.sum(p)}")
    
def chi_square(img, h, w):
    _, counts = np.unique(img, return_counts=True)
    
    prob = (counts / (h * w)) - 1/256
    temp = np.sum(np.power(prob, 2))
    
    x_2 = np.sqrt(256 * h * w * temp)
    print(f"The chisquare is: {x_2}")
    
def npcr(img1, img2, h, w):
    res = (np.count_nonzero(np.equal(img1, img2) == 0)) / (h * w)
    print(f'The NPCR is: {res}')
    
def uaci(img1, img2, h, w):
    res = np.sum(np.abs(img1 - img2) / 255) / (h * w)
    print(f'The UACI is: {res}')

def parser_arguments():
    parser = argparse.ArgumentParser(prog='RDHEI-EMR-LMR',
                                     usage="%(prog)s [-h] [method: type 'EMR' or 'LMR']",
                                     description='Reversible data hiding in encrypted images')

    parser.add_argument('method',
                        metavar='method',
                        type=str,
                        help="type 'EMR' or 'LMR'")
    # parser.add_argument('image path', type=str, help='the test image path')

    args = parser.parse_args()
    return args.method.upper()


if __name__ == '__main__':

    method = parser_arguments()

    image_path = 'assets/images/lena.pgm'
    img = skimage.io.imread(image_path)
    original_img = img.copy()
    h, w = original_img.shape

    # if method == 'EMR':
    #     content_owner = EMR.EMRContentOwner()
    #     data_hider = EMR.EMRDataHider()
    #     recipient = EMR.EMRRecipient()
    # elif method == 'LMR':
    #     content_owner = LMR.LMRContentOwner()
    #     data_hider = LMR.LMRDataHider()
    #     recipient = LMR.LMRRecipient()
    
    content_owner = EMR.EMRContentOwner()
    data_hider = EMR.EMRDataHider()
    recipient = EMR.EMRRecipient()
    
    # Perform the corresponding method based on the user input
    # Construct the RRBE scheme
    # Generate the secret key
    secret_key = utils.crypto_tools.generate_secret_key_1(*img.shape)

    encoded_img, encrypt_img, msb = content_owner.encode_image(img, secret_key).values()
    
    marked_encoded_img, secret_key_2, msb, info = data_hider.hiding_data(encoded_img, msb).values()
    marked_encrypted_img = marked_encoded_img.copy()
    message = recipient.extract_message(marked_encoded_img, secret_key_2, msb)
    print((info == message).all())
    
    recovered_img = recipient.recover_image(marked_encoded_img, secret_key, msb)
    
    # Caculate the PSNR
    psnr = metrics.peak_signal_noise_ratio(original_img, recovered_img, data_range=None)
    print(f"The Peak Signal-to-Noise Ratio is: {psnr}")
    
    # Calculate the SSIM
    ssim = metrics.structural_similarity(original_img, recovered_img, data_range=recovered_img)
    print(f"The Structural SIMilarity is: {ssim}")
    
    # Calculate the Shannon Entropy
    # shannon_entropy(encrypt_img, h, w)
    # chi_square(encrypt_img, h, w)
    
    shannon_entropy(marked_encrypted_img, h, w)
    chi_square(marked_encrypted_img, h, w)
    
    npcr(original_img, encrypt_img, h, w)
    uaci(original_img, encrypt_img, h, w)