# Searching-Eclipsing-Binaries-with-TESS
Using the TESS 30-min interval full frame images, this code searches for eclipsing binaries with convolutional neural network (CNN). 

<!--  -->
![10,60  1 0000](https://user-images.githubusercontent.com/49893001/94355126-41f8f200-0036-11eb-89fa-31997ef65cc8.png)

## Getting Started

This python script can be used to search for EBs in a cluster or any region of the sky covered by TESS. It returns a graph and .dat document for each pixel, which include information about the CNN prediction on the light curve of the pixel. 

### Prerequisites

Packages needed to run this script include
```
pickle, tensorflow, sklearn, astroquery
```

### Installing
After installing required packages, clone the master folder and run Searching_EB-Command_line.py on a command line. Locate tess_cnn.h5 file in CNN Training folder and you can start using it!

## Running the Search
The script askes for several parameters and saves figures and data that are selected by these parameters. The time taken for each pixel is roughly 6 seconds. Below is an example for the open cluster [FSR2007] 0728, which returns the figure shown above. This is a 400 pixel cut, and it took approximately 40 minutes in a test run. 

```
Target Identifier: [FSR2007] 0728
FOV in arcmin (max 33) [Default: 5]: 6.7
Trained cnn .h5 file [Default: tess_cnn.h5]:
#################################
  sectorName   sector camera ccd
-------------- ------ ------ ---
tess-s0019-1-3     19      1   3
#################################
Please choose a threshold to save a figure. This number is the prediction given by cnn, 
indicating the likelihood of including eclipsing binaries in a light curve. 0 is the 
lowest (keeping every pixel), and 1 is the highest (maybe excluding everything).
Threshold [Default: 0.95]: 0.99
Saving figures to [Default: root]:
Finished pixels:  ███████████████████∙∙∙∙∙∙∙∙∙∙∙∙∙ 60.0% Elapsed: 1514s Remaining: 1003s
```
Change the target and size of the cut to test it on any target. Note: Each TESS pixel is about 21 arcsec wide.

## Log
* 8.24.2020: Created Rrepository, uploaded CNN training and Searching EBs folder.
* 8.29.2020: Created .py file to be run in the command line.
* 9.15.2020: Added detrending function (Wotan).
* 9.26.2020: Updated cnn model (tess_cnn_weights.h5) by using real light curves of TESS.

## Contributers

* **Te Han** 
* **Timothy D Brandt** 

## License
 ...
