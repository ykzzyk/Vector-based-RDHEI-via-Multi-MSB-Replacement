import argparse
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import math

import EMR
import LMR
import utils

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
    # Generate the secret key
    secret_key = utils.crypto_tools.generate_secret_key_1(*img.shape)

    encoded_img, encrypt_img, msb = content_owner.encode_image(img, secret_key).values()
    
    marked_encoded_img, secret_key_2, msb, info = data_hider.hiding_data(encoded_img, msb).values()

    message = recipient.extract_message(marked_encoded_img, secret_key_2, msb)
    print((info == message).all())
    
    recovered_img = recipient.recover_image(marked_encoded_img, secret_key, msb)
    
    fig = plt.figure(1, figsize=(8,8))
    axis1 = plt.subplot(221)
    plt.imshow(img)
    axis1.set_title('Original Image')

    axis2 = plt.subplot(222)
    plt.imshow(encrypt_img)
    axis2.set_title('Encrypted Image')
    
    axis3 = plt.subplot(223)
    plt.imshow(marked_encoded_img)
    axis3.set_title('Marked Encrypted Image')
    
    axis4 = plt.subplot(224)
    plt.imshow(recovered_img)
    axis4.set_title('Reconstructed Image')
    
    plt.show()