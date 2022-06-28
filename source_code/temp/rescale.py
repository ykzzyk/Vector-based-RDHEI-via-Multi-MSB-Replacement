import matplotlib.pyplot as plt
import skimage
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean

# def parser_arguments():
#     parser = argparse.ArgumentParser(prog='RDHEI-EMR-LMR',
#                                      usage="%(prog)s [-h] [method: type 'EMR' or 'LMR']",
#                                      description='Reversible data hiding in encrypted images')

#     parser.add_argument('method',
#                         metavar='method',
#                         type=str,
#                         help="type 'EMR' or 'LMR'")
    
#     parser.add_argument('name', 
#                         metavar='name', 
#                         type=str, 
#                         help='The test image name')


#     args = parser.parse_args()

image_path = f'assets/images/head.pgm'
img = skimage.io.imread(image_path)
h, w = img.shape
print(h,w)

for i in [0.25, 0.5, 0.75]:
    image = rescale(img, i, anti_aliasing=True)
    skimage.io.imsave(f'assets/images/head_{int(i*h)}.pgm', image)