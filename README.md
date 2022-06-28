# Vector-based RDHEI via Multi-MSB Replacement

This is a project for IEEE TCSVT journal paper [vector-based reversible data hiding in encrypted images via multi-MSB replacement](https://ieeexplore.ieee.org/document/9796573/authors#authors).

## Getting Started

These instructions below will guide you on running this project on your local machine for development and testing purposes.

> **Note** For addtional information and references, please check the [Appendix](http://dp.stmarytx.edu/wluo/Appendix.pdf) for this paper.

> **Warning** Our code can run in MAC/Linux system, but still have some errors when running in Windows.

### File structure

There is the file structure provided below:

```
RDHEI-EMR-LMR
	└── source_code
			└── assets // for all testing images
			└── outputs // for all the testing outputs
			└── temp // store temporary files, not important
			└── tools // JBIG-KIT excutable software
			└── utils
			└── auto.py // for automated testing of 10,000 images in BOWS2 database
			└── demo.py // for demo a specific image purpose
			└── eval.py // for security and image resolution evaluation purpose
			└── entity.py
			└── EMR.py
			└── LMR.py

	└── environemnt.yml // use conda/miniconda to recreate the environment
	└── LICENSE
	└── README.md
	└── requirements.txt // use pip to recreate the enviroment
```

### Prerequisites

We recommend to use **conda** or **miniconda** to install the environments. The install comments are listed as following:

```
# Create an enviroment
conda create --name test python=3.8
# Activate the new enviroment
conda activate test
# Install relate packages
conda install argparse
python -m pip install scikit-image # will install nump
pip install pandas # For testing the auto.py file
pip install tdqm # For testing the auto.py file
conda install -c conda-forge matplotlib # For visualization
pip install pathlib
```

But if you prefer to use **pip** install the required enviroment for this project, there is also a file named "requirements.txt" to help.

```
# An Example
# Use conda/miniconda to create the same environment to run this project
# Create the environment from the environment.yml file

conda env create -f environment.yml

```

## Running the tests

There are three files under the source_code dictionary for testing purpose: auto.py, demo.py, eval.py

### Run the demo.py file

This is a demo for using EMR or LMR to test a specific image:

```
cd source_code
python demo.py <method> <image_name>

# for example, we test image lena use EMR method:
python demo.py EMR lena
```

The demo result is stored in the outputs dictionary, and the file name is "EMR_demo_lena.png".

**The demo tesing result of image lena with the EMR method:**

![EMR_demo_lena](https://user-images.githubusercontent.com/55161270/120936049-69c34e00-c6cb-11eb-9928-56488ba3124e.png)

```
# for example, we test image lena use LMR method:
python demo.py LMR lena
```

The demo result is stored in the outputs dictionary, and the file name is "LMR_demo_lena.png".

**The demo tesing result of image lena with the LMR method:**
![LMR_demo_lena](https://user-images.githubusercontent.com/55161270/120936135-e2c2a580-c6cb-11eb-8a00-a0597a3a2037.png)

### Run the eval.py file

This is a evaluation for using EMR or LMR to test a specific image:

```
cd source_code
python eval.py <method> <image_name>

# for example, we test image lena use EMR method:
python eval.py EMR lena
```

**The evaluation tesing result of image lena with the EMR method:**

```
----- Test the image lena -----

----- Maximum Data Embedding Rate -----
The maximum data embedding rate (DER) is: 2.65667724609375

----- Secret Information Extraction Phase -----
Is it error-free? Answer: True

----- Shannon Entropy Results -----
The Shannon Entropy of the original image is: 7.44556757034006
The Shannon Entropy of the encrypted image is: 7.999250158874972
The Shannon Entropy of the marked encrypted image is: 7.919896506351618

----- Chi Square Results -----
The chisquare of the original image is: 340.6469861023652
The chisquare of the encrypted image is: 16.507810651325027
The chisquare of the marked encrypted image is: 169.09512739690342

----- NPCR Results -----
The NPCR between original image and encrypted image is: 0.9960899353027344
The NPCR between original image and marked encrypted image is: 0.9959945678710938

----- UACI Results -----
The UACI between original image and encrypted image is: 0.28583332136565565
The UACI between original image and marked encrypted image is: 0.2866824729769837

----- PSNR Results -----
The Peak Signal-to-Noise Ratio between original image and encrypted image is: 9.239278996208478
The Peak Signal-to-Noise Ratio between original image and marked encrypted image is: 9.222844749731657
The Peak Signal-to-Noise Ratio between original image and recovered image is: 51.143920872280816

----- SSIM Results -----
The Structural SIMilarity between original image and recovered image is: 0.996270899634543
```

```
# for example, we test image lena use EMR method:
python eval.py LMR lena
```

**The evaluation tesing result of image lena with the LMR method:**

```
----- Test the image lena -----

----- Maximum Data Embedding Rate -----
The maximum data embedding rate (DER) is: 1.9925079345703125

----- Secret Information Extraction Phase -----
Is it error-free? Answer: True

----- Shannon Entropy Results -----
The Shannon Entropy of the original image is: 7.44556757034006
The Shannon Entropy of the encrypted image is: 7.999194930860796
The Shannon Entropy of the marked encrypted image is: 7.999162516610978

----- Chi Square Results -----
The chisquare of the original image is: 340.6469861023652
The chisquare of the encrypted image is: 17.095035372148253
The chisquare of the marked encrypted image is: 17.422428522596956

----- NPCR Results -----
The NPCR between original image and encrypted image is: 0.99609375
The NPCR between original image and marked encrypted image is: 0.9959564208984375

----- UACI Results -----
The UACI between original image and encrypted image is: 0.2866433237113205
The UACI between original image and marked encrypted image is: 0.2861614750880821

----- PSNR Results -----
The Peak Signal-to-Noise Ratio between original image and encrypted image is: 9.218278226235427
The Peak Signal-to-Noise Ratio between original image and marked encrypted image is: 9.235855003727162
The Peak Signal-to-Noise Ratio between original image and recovered image is: inf

----- SSIM Results -----
The Structural SIMilarity between original image and recovered image is: 1.0
```

### Run the auto.py file

This file is for automated testing purpose:

```
cd source_code
python auto.py <method> <handle> <images> # <method>: emr/lmr, <handle>: test/open, <images>: how many images will be tested/opened.
```

We stored the auto testing result of 10,000 imaegs using EMR method in the outputs dictionary, and the file name is "EMR_10000.csv".

```
# for example, if we want to open the EMR_10000.csv file

python auto.py emr open 10000
```

**The auto tesing result of image lena with the EMR method:**

```
################### DER Analysis ###################


----- The Maximum DER is: ----
6.268180847167969

----- The Minimal DER is: ----
1.2087364196777344

----- The Average DER is: ----
3.2456654582977293


################### MSB Analysis ###################


----- The Maximum MSB is: ----
7

----- The Minimal MSB is: ----
2

----- The Most Frequent MSB is: ----
5


################### PSNR Analysis ###################


----- The Maximum PSNR is: ----
51.17239515656845

----- The Minimal PSNR is: ----
51.110891251462775

----- The Average PSNR is: ----
51.1409200454349


################### SSIM Analysis ###################


----- The Maximum SSIM is: ----
0.9997702176548636

----- The Minimal SSIM is: ----
0.974620836378378

----- The Average SSIM is: ----
0.995820349480765
```

We stored the auto testing result of 10,000 imaegs using LMR method in the outputs dictionary, and the file name is "LMR_10000.csv".

```
# for example, if we want to open the LMR_10000.csv file

python auto.py lmr open 10000
```

**The auto tesing result of image lena with the LMR method:**

```

################### DER Analysis ###################


----- The Maximum DER is: ----
6.016880035400391

----- The Minimal DER is: ----
0.21290206909179688

----- The Average DER is: ----
2.532548918533325


################### MSB Analysis ###################


----- The Maximum MSB is: ----
8

----- The Minimal MSB is: ----
2

----- The Most Frequent MSB is: ----
6


################### PSNR Analysis ###################


----- The Maximum PSNR is: ----
inf

----- The Minimal PSNR is: ----
inf

----- The Average PSNR is: ----
inf

################### SSIM Analysis ###################


----- The Maximum SSIM is: ----
1

----- The Minimal SSIM is: ----
1

----- The Average SSIM is: ----
1
```

## Authors

- **Yike Zhang** - _Initial work_ - [GitHub](https://github.com/ykzzyk/RDHEI-EMR-LMR)

## License

This project is licensed under the MIT License.
