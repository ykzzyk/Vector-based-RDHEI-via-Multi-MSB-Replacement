import argparse
import skimage.io
import os
import matplotlib.pyplot as plt
import numpy as np
import math

import EMR
import LMR
import utils
from pathlib import Path
import tqdm
import csv
import pickle
import warnings
from skimage import metrics
from collections import Counter

def parser_arguments():
    parser = argparse.ArgumentParser(prog='RDHEI-EMR-LMR',
                                     usage="%(prog)s [-h] [method: type 'EMR' or 'LMR']",
                                     description='Reversible data hiding in encrypted images')

    parser.add_argument('method',
                        metavar='method',
                        type=str,
                        help="type 'EMR' or 'LMR'")    
    
    parser.add_argument('handle',
                        metavar='handle',
                        type=str,
                        help="Please type how do you want to handle the files? 'test' or 'open' ")
    
    parser.add_argument('images',
                        metavar='images',
                        type=int,
                        help="Please type how many images needed to be tested?")

    args = parser.parse_args()
    
    return args.method.upper(), args.handle.upper(), args.images

if __name__ == "__main__":
    
    dir_sum = {
        "image_name": [],
        "DER": [],
        "MSB": [],
        "PSNR": [],
        "SSIM": []
    }
    
    lmr_bad_cases = 0
    
    method, file_handle, num = parser_arguments()
    
    if file_handle == 'TEST':
        start = 3000
        for i in tqdm.tqdm(range(start, num)):
            image_path = Path.cwd() / 'assets' / 'BOWS2' / f'{i}.pgm'
            img = skimage.io.imread(image_path)
            h, w = img.shape

            if method == 'EMR':
                content_owner = EMR.EMRContentOwner()
                recipient = EMR.EMRRecipient()
            elif method == 'LMR':
                content_owner = LMR.LMRContentOwner()
                
            
            secret_key_1 = utils.crypto_tools.generate_secret_key_1(*img.shape)

            encoded_img, encrypt_img, msb, der = content_owner.encode_image(img, secret_key_1).values()
    
            if msb == None:
                dir_sum["image_name"].append(f'{i}.pgm')
                dir_sum["DER"].append('None')
                dir_sum["MSB"].append('None')
                dir_sum["PSNR"].append('None')
                dir_sum["SSIM"].append('None')  
                lmr_bad_cases += 1
                continue
            
            dir_sum["image_name"].append(f'{i}.pgm')
            dir_sum["DER"].append(der)
            dir_sum["MSB"].append(msb)
            
            if method == 'EMR':
            
                recovered_img = recipient.recover_image(encoded_img, secret_key_1, msb)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    psnr = metrics.peak_signal_noise_ratio(img, recovered_img, data_range=None)
                    
                ssim = metrics.structural_similarity(img, recovered_img, data_range=recovered_img)

                dir_sum["PSNR"].append(psnr)
                dir_sum["SSIM"].append(ssim)  
                
            elif method == 'LMR':
                
                dir_sum["PSNR"].append('inf')
                dir_sum["SSIM"].append(1)  
            
        with open(Path.cwd() / 'outputs' / f'{method}_{num}.pickle', 'ab+') as handle:
            pickle.dump(dir_sum, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(Path.cwd() / 'outputs' / f'{method}_{num}.csv', 'a+') as f:        
            writer = csv.writer(f)
            if start == 0:
                writer.writerow(dir_sum.keys())
            writer.writerows(zip(*dir_sum.values()))
          
        print("\n----- Good cases are: -----")  
        print((num - lmr_bad_cases) / num)
        
        print("\n----- Bad cases are: -----")  
        print((lmr_bad_cases) / num)
        print("\n")
            
    elif file_handle == 'OPEN':
        
        try:
            with open(Path.cwd() / 'outputs' / f'{method}_{num}.pickle', 'rb') as handle:
                dir_sum = pickle.load(handle)                
                
            print("\n\n################### DER Analysis ###################\n")
    
            print("\n----- The Maximum DER is: ----")
            print(max(dir_sum["DER"]))
            
            print("\n----- The Minimal DER is: ----")
            print(min(dir_sum["DER"]))
            
            print("\n----- The Average DER is: ----")
            print(sum(dir_sum["DER"]) / num)
            
            print("\n\n################### MSB Analysis ###################\n")
            
            print("\n----- The Maximum MSB is: ----")
            print(max(dir_sum["MSB"]))
            
            print("\n----- The Minimal MSB is: ----")
            print(min(dir_sum["MSB"]))
            
            print("\n----- The Most Frequent MSB is: ----")
            print((Counter(dir_sum["MSB"])).most_common(1)[0][0])
            
            print("\n\n################### PSNR Analysis ###################\n")
            
            if method == 'EMR':
                
                print("\n----- The Maximum PSNR is: ----")
                print(max(dir_sum["PSNR"]))
                
                print("\n----- The Minimal PSNR is: ----")
                print(min(dir_sum["PSNR"]))
                
                print("\n----- The Average PSNR is: ----")
                print(sum(dir_sum["PSNR"]) / num)
                
                print("\n\n################### SSIM Analysis ###################\n")
                
                print("\n----- The Maximum SSIM is: ----")
                print(max(dir_sum["SSIM"]))
                
                print("\n----- The Minimal SSIM is: ----")
                print(min(dir_sum["SSIM"]))
                
                print("\n----- The Average SSIM is: ----")
                print(sum(dir_sum["SSIM"]) / num)
                print("\n")
                
            elif method == 'LMR':
                print("\n----- The Maximum PSNR is: ----")
                print("inf")
                
                print("\n----- The Minimal PSNR is: ----")
                print("inf")
                
                print("\n----- The Average PSNR is: ----")
                print("inf")
                
                print("\n################### SSIM Analysis ###################\n")
                
                print("\n----- The Maximum SSIM is: ----")
                print(1)
                
                print("\n----- The Minimal SSIM is: ----")
                print(1)
                
                print("\n----- The Average SSIM is: ----")
                print(1)
                print("\n")
                
        except FileNotFoundError:
            print("Sorry, the file does not exsit")