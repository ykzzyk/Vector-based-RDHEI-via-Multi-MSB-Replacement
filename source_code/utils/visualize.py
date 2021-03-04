
import matplotlib.pyplot as plt
import numpy as np

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