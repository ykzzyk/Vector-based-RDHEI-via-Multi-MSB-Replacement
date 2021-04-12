import argparse
import skimage.io
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
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

def plot_der(x_axis, y_axis, average_axis, x_lim, method):
    my_figure = plt.figure(figsize=[30, 10])
    ax = my_figure.add_subplot(1, 1, 1)

    # Removing top and right spines
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)

    # Re-directioning ticks inward
    ax.tick_params(direction="in")

    # plt.scatter(x_axis, y_axis, s=15, c="navy")
    plt.plot(x_axis, y_axis, c="blue", linewidth=1, marker='.', fillstyle='none')
    plt.plot(x_axis, average_axis, color="red", label='Average', linewidth=2)

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)

    ax.xaxis.set_tick_params(width=3, length=13)
    ax.yaxis.set_tick_params(width=3, length=13)

    yticks = ax.yaxis.get_major_ticks()
    yticks[0].set_visible(False)

    font = font_manager.FontProperties(size=35)
    plt.legend(loc='best', prop=font, edgecolor='black')

    plt.xlabel('images', fontsize=35)
    plt.ylabel('DER (bpp)', fontsize=35)

    plt.xlim(0, x_lim-1)
    plt.ylim(0, 6)

    filepath = Path.cwd() / 'outputs' / f'{method}_der_plot_{x_lim}'

    plt.savefig(filepath, dpi=200, bbox_inches='tight')


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
            df = pd.read_csv(Path.cwd() / 'outputs' / f'{method}_{num}.csv')         
                
            print("\n\n################### DER Analysis ###################\n")
    
            print("\n----- The Maximum DER is: ----")
            print(max(df["DER"]))
            
            print("\n----- The Minimal DER is: ----")
            print(min(df["DER"]))
            
            print("\n----- The Average DER is: ----")
            average_der = sum(df["DER"]) / num
            print(average_der)
            
            print("\n\n################### MSB Analysis ###################\n")
            
            print("\n----- The Maximum MSB is: ----")
            print(max(df["MSB"]))
            
            print("\n----- The Minimal MSB is: ----")
            print(min(df["MSB"]))
            
            print("\n----- The Most Frequent MSB is: ----")
            print((Counter(df["MSB"])).most_common(1)[0][0])
            
            print("\n\n################### PSNR Analysis ###################\n")
            
            if method == 'EMR':
                
                print("\n----- The Maximum PSNR is: ----")
                print(max(df["PSNR"]))
                
                print("\n----- The Minimal PSNR is: ----")
                print(min(df["PSNR"]))
                
                print("\n----- The Average PSNR is: ----")
                print(sum(df["PSNR"]) / num)
                
                print("\n\n################### SSIM Analysis ###################\n")
                
                print("\n----- The Maximum SSIM is: ----")
                print(max(df["SSIM"]))
                
                print("\n----- The Minimal SSIM is: ----")
                print(min(df["SSIM"]))
                
                print("\n----- The Average SSIM is: ----")
                print(sum(df["SSIM"]) / num)
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

            # Plot the DER
            plot_der(
                [i for i in range(len(df['DER']))], 
                df['DER'], 
                [average_der for i in range(len(df['DER']))],
                num,
                method
            )

                
        except FileNotFoundError:
            print("Sorry, the file does not exsit")