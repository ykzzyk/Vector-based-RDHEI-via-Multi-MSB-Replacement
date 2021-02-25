import abc
import numpy as np

# Local imports
import new_utility as utils

# This code should reflect the global scheme of the EMR and LMR

class ContentOwner(abc.ABC):
    
    def encode(self, img: np.ndarray, secret_key: np.ndarray) -> np.ndarray:
        
        # Map generation
        generated_maps = self.map_generation(img)

        # Encryption
        encrypted_img = utils.encrypt_img(img, secret_key)

        # Map Embedding
        encoded_img = self.embed_img(encrypted_img, generated_maps)

        # Returning encoded (encrypted and embedded) image 
        return encoded_img

    @abc.abstractmethod
    def map_generation(self, img: np.ndarray) -> dict:
        ...

    @abc.abstractmethod
    def embed_img(self, img: np.ndarray, maps: dict) -> np.ndarray:
        ...

    def generate_location_map(self, img: np.ndarray, msb: np.ndarray) -> np.ndarray:
        """
        Generates all possible location maps and determines the best one based
        on its bpp.
        """

        h, w = img.shape
        msbs = msb.shape[0]

        # Expand image to account for multiple location maps
        e_img = np.expand_dims(img, axis=0)

        # Create a 8-bit template of MSB 1s followed by MSB 0s
        scalar_mask = ((1 << msb) - 1) << (8 - msb)
        mask = np.resize(scalar_mask,  (h, msbs)).T
        past_data = np.zeros_like(mask)

        # Create location map
        lms = np.zeros((msbs, h, w), dtype='uint8')

        # Set initial condition
        lms[..., 0] = 1
        past_data[..., :] = e_img[..., 0]

        # Image processing
        for i in range(1,w):

            msb_changed = (e_img[..., i] & mask != past_data & mask)
            past_data = np.where(msb_changed, e_img[...,i], past_data)
            lms[msb_changed, i] = 1

        # Select the best location map
        _, h, w = lms.shape

        # Calculate the bpp for all and select the best
        bpps = (msb / (h * w)) * np.sum(lms == 0, axis=(1,2))
        max_index = np.argmax(bpps)
        #max_bpp = bpps[max_index]
        #max_msb = msb[max_index]
        max_lm = lms[max_index]

        return max_lm

    def original_generate_location_map(self, img, msb):
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
    
class DataHider(abc.ABC):
    
    @abc.abstractmethod
    def hiding_data(self, img: np.ndarray) -> np.ndarray:
        ...
    
class Recipient(abc.ABC):
    
    @abc.abstractmethod
    def recover_image(self, img: np.ndarray) -> np.ndarray:
        ...
          
    @abc.abstractmethod
    def extract_message(self, img: np.ndarray) -> np.ndarray:
        ...