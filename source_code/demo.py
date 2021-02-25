import EMR
import LMR
import argparse


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

    image_path = 'Assets/Sample/peppers.pgm'

    if method == 'EMR':
        content_owner = EMR.ContentOwner()
        data_hider = EMR.DataHider()
        recipient = EMR.Recipient()
    elif method == 'LMR':
        content_owner = LMR.ContentOwner()
        data_hider = LMR.DataHider()
        recipient = LMR.Recipient()

    # Perform the corresponding method
    output = content_owner.preprocess_image(image_path)
    # output = dh.asdf(output)
    # final_output = rp.asdf(output)
