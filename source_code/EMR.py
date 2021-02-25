import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import PIL

# matplotlib.use("Qt5Agg")

# Local Imports
import entity
import new_utility as utils

class EMRContentOwner(entity.ContentOwner):

    MSBS = np.array([2,3,4,5,6,7])

    def __init__(self):
        super(EMRContentOwner, self).__init__()

    def map_generation(self, img: np.ndarray) -> dict:
        
        # Construct the best location map
        lm1 = self.generate_location_map(img, self.MSBS)
        lm2 = self.original_generate_location_map(img, np.array([4]))
        
        plt.imshow(lm1)
        plt.show()

        # return a dictionary
        return {'location_map': lm2}

    def embed_img(self, img: np.ndarray, maps: dict) -> np.ndarray:
        return np.array([0])
        
class EMRDataHider(entity.DataHider):
    
    def __init__(self):
        super(EMRDataHider, self).__init__()
        
    def hiding_data(self, img: np.ndarray) -> np.ndarray:
        return None
        
    
class EMRRecipient(entity.Recipient):
    
    def __init__(self):
        super(EMRRecipient, self).__init__()
        
    def recover_image(self, img: np.ndarray) -> np.ndarray:
        return None

    def extract_message(self, img: np.ndarray) -> np.ndarray:
        return None

if __name__ == '__main__':

    # Debugging EMR

    # Initializing all the participants
    co = EMRContentOwner()
    dh = EMRDataHider()
    rp = EMRRecipient()

    # Load test image and create random encryption key
    img = skimage.io.imread('assets/images/peppers.pgm')
    #img = skimage.io.imread('peppers.pgm')
    plt.imshow(img); plt.show()
    
    #secret_key = utils.generate_secret_key(*img.shape)
    
    # Construct the RRBE scheme
    #encoded_img = co.encode(img, secret_key)
        