import argparse
import skimage.io
import matplotlib.pyplot as plt

import EMR
import LMR


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

    image_path = 'assets/images/baboon.pgm'
    img = skimage.io.imread(image_path)

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
    encoded_img, secret_key, msb = content_owner.encode_image(img).values()
    marked_encoded_img, msb = data_hider.hiding_data(encoded_img, msb).values()
    message = recipient.extract_message(marked_encoded_img, msb)
    recovered_img = recipient.recover_image(marked_encoded_img, secret_key, msb)

    # Show the recovered image
    plt.imshow(recovered_img)
    plt.show()
    
