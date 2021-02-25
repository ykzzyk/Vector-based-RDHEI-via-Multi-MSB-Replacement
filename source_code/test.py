from PIL import Image  # Import Image from Pillow mudule
import numpy as np  # Import Numpy
import time
import pdb
import matplotlib.pyplot as plt

test_image = 'Assets/Sample/peppers.pgm'


def save_img(img, name='test'):
    fig = plt.figure()
    plt.imshow(img)
    fig.savefig(f'{name}.png', dpi=300)


def make_summary_figure(**images):
    # Calculating the number of rows and columns
    nr = len(images)
    nc = images[list(images.keys())[0]].shape[0]

    h = nr * images[list(images.keys())[0]][0].shape[0]
    w = nc * images[list(images.keys())[0]][0].shape[1]

    largest_dim = max(h, w)
    largest_size_in_inches = 8

    h_ratio = h / largest_dim
    w_ratio = w / largest_dim

    cal_h = h_ratio * largest_size_in_inches
    cal_w = w_ratio * largest_size_in_inches

    # Initializing the figure and axs
    fig = plt.figure(figsize=(cal_w, cal_h))
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=0,
        hspace=0
    )

    for i, (name, image) in enumerate(images.items()):
        if len(image.shape) >= 3:  # NHW or NCHW
            for j, img in enumerate(image):

                plt.subplot(nr, nc, 1 + j + nc * i)
                plt.xticks([])
                plt.yticks([])
                if j == 0:
                    plt.ylabel(' '.join(name.split('_')).title())
                    plt.text(25, 25, ' '.join(name.split('_')).title(), bbox=dict(facecolor='white', alpha=0.5),
                             color=(0, 0, 0))
                plt.imshow(img)
        else:  # HW only

            plt.subplot(nr, nc, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)

    return fig


def original_generate_location_map(img, msb):
    """RRBE Schema: Process Image to reserve room for data hiding"""
    h, w = img.shape

    # Generate template
    template = ((1 << msb) - 1) << (8 - msb)

    # Create location map
    lm = np.zeros_like(img, dtype='uint8')

    # Image processing
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                # Mark the first pixel in the location map: "1"
                lm[i, j] = 1
                mark = img[i, j]
                continue
            # MSB doesn't match
            if img[i, j] & template != mark & template:
                # Mark the pixels in the location map: "1"
                lm[i, j] = 1
                mark = img[i, j]

    # Return location map
    return lm


def new_generate_location_map(img, msb):
    """RRBE Schema: Process Image to reserve room for data hiding"""
    h, w = img.shape

    # Create a 8-bit template of MSB 1s followed by MSB 0s
    mask = np.resize(np.array([((1 << msb) - 1) << (8 - msb)]), (h,))
    past_data = np.zeros_like(mask)

    # Create location map
    lm = np.zeros_like(img, dtype='uint8')

    # Set initial condition
    lm[:, 0] = 1
    past_data = img[:, 0]

    # Image processing
    for i in range(1, w):
        msb_changed = (img[:, i] & mask != past_data & mask)
        past_data = np.where(msb_changed, img[:, i], past_data)
        lm[msb_changed, i] = 1

    return lm


def generate_best_location_map(
        img: np.ndarray,
        msb: np.ndarray = np.array([2, 3, 4, 5, 6, 7])
):
    h, w = img.shape
    msbs = msb.shape[0]

    # Expand image to account for multiple location maps
    e_img = np.expand_dims(img, axis=0)

    # Create a 8-bit template of MSB 1s followed by MSB 0s
    scalar_mask = ((1 << msb) - 1) << (8 - msb)
    mask = np.resize(scalar_mask, (h, msbs)).T
    past_data = np.zeros_like(mask)

    # Create location map
    lms = np.zeros((msbs, h, w), dtype='uint8')

    # Set initial condition
    lms[..., 0] = 1
    past_data[..., :] = e_img[..., 0]

    # Image processing
    for i in range(1, w):
        msb_changed = (e_img[..., i] & mask != past_data & mask)
        past_data = np.where(msb_changed, e_img[..., i], past_data)
        lms[msb_changed, i] = 1

    return lms


def calculate_bpp(lm, msb):
    # Determining the size of the location map
    h, w = lm.shape

    # Calculating the bbp
    bpp = (msb / (h * w)) * np.sum(lm == 0)  # Calculate the Payload
    return bpp


def get_best_lm(lms, msb):
    _, h, w = lms.shape

    # Calculate the bpp for all and select the best
    bpps = (msb / (h * w)) * np.sum(lms == 0, axis=(1, 2))

    max_msb_index = np.argmax(bpps)

    max_bpp = bpps[max_msb_index]
    max_msb = msb[max_msb_index]
    max_lm = lms[max_msb_index]

    return max_msb, max_lm, max_bpp


if __name__ == '__main__':

    # Loading image
    img = np.array(Image.open(test_image))

    # Trim the image to ease debugging
    # img = img[:5, :5]

    total_o_ms_diff = 0
    total_n_ms_diff = 0

    # For all possible values of msb
    # """
    for msb in [2, 3, 4, 5, 6, 7]:
        # Generate location maps
        tic = time.time()
        original_lm = original_generate_location_map(img, msb)
        toc = time.time()
        original_ms_diff = (toc - tic) * 1000
        total_o_ms_diff += original_ms_diff

        tic = time.time()
        new_lm = new_generate_location_map(img, msb)
        toc = time.time()
        new_ms_diff = (toc - tic) * 1000
        total_n_ms_diff += new_ms_diff

        save_img(original_lm, f'original_{msb}.png')
        save_img(new_lm, f'new_{msb}.png')

        # Calculate the bpp for both location maps
        original_bpp = calculate_bpp(original_lm, msb)
        new_bpp = calculate_bpp(new_lm, msb)

        # Reports the results
        print(f'msb={msb}')
        print(f'\tOriginal: bpp={original_bpp:.5f} time={original_ms_diff:.5f}')
        print(f'\t     New: bpp={new_bpp:.5f} time={new_ms_diff:.5f}')
    # """

    print("\nOptimized best location map retrival")
    msb = np.array([2, 3, 4, 5, 6, 7])
    tic = time.time()
    lms = generate_best_location_map(img, msb)
    toc = time.time()
    all_n_ms_diff = (toc - tic) * 1000

    best_msb, best_lm, best_bpp = get_best_lm(lms, msb)
    print(f'Best msb = {best_msb} with bpp = {best_bpp:.5f}')

    print("\nComparision of all methods:")
    print(f"Total Original time: {total_o_ms_diff:.5f}")
    print(f"Total New      time: {total_n_ms_diff:.5f}")
    print(f"Total New++    time: {all_n_ms_diff:.5f}")