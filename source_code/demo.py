import argparse
import skimage.io
import matplotlib.pyplot as plt
from skimage import metrics

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
    original_img = img.copy()

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
    secret_key = utils.crypto_tools.generate_secret_key(*img.shape)

    encoded_img, msb = content_owner.encode_image(img, secret_key).values()
    marked_encoded_img, msb = data_hider.hiding_data(encoded_img, msb).values()
    message = recipient.extract_message(marked_encoded_img, msb)
    recovered_img = recipient.recover_image(marked_encoded_img, secret_key, msb)
    
    # Caculate the PSNR
    psnr = metrics.peak_signal_noise_ratio(original_img, recovered_img, data_range=None)
    print(f"The Peak Signal-to-Noise Ratio is: {psnr}")
    
    # Calculate the SSIM
    ssim = metrics.structural_similarity(original_img, recovered_img, data_range=recovered_img)
    print(f"The Structural SIMilarity is: {ssim}")
