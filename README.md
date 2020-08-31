# Searching-Eclipsing-Binaries-with-TESS
Using the TESS 30-min interval full frame images, this code searches for eclipsing binaries with convolutional neural network (CNN). 

<!-- ![NGC659_pixel-12-11](https://user-images.githubusercontent.com/49893001/91348678-fdf5a180-e798-11ea-8324-f04a22370dd7.png)
![NGC_659](https://user-images.githubusercontent.com/49893001/91349614-685b1180-e79a-11ea-8acf-429b8d7b252a.png) -->

## Getting Started

This python script can be used to search for EBs in a cluster or any region of the sky covered by TESS. It returns a graph and .dat document for each pixel that includes information of this CNN prediction

### Prerequisites

Packages needed to run this script includes 
```
pickle, tensorflow, sklearn, astroquery
```

### Installing
Clone the master folder and run Searching_EB-Command_line.py on a command line. Locate tess_cnn.h5 file in CNN Training folder and you can start using it!

## Running the tests
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


## Authors

* **Te Han** 

## License
 ...
