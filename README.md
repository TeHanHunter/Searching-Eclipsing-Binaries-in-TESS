# Searching-Eclipsing-Binaries-with-TESS
Using the TESS 30-min interval full frame images, this code searches for eclipsing binaries with convolutional neural network (CNN). With `Batman`, the code produces pseudo light curves of EBs and trains a CNN model with them. This model is then used on each pixel of full frame images produced by TESScut. The output includes all light curves with high value cnn predictions above an arbitrary threshold. The image below is a sample result. 

<!--  -->
![10,60  1 0000](https://user-images.githubusercontent.com/49893001/97091618-0a0fab00-15f2-11eb-926e-097c558bb119.png)

The periods to test cnn on is carefully chosen to avoid huge time intervals between adjacent data points, which make interpolation do a poor job. The below image shows the difference of standard deviation of time intervals between a blunt choice (geometric series) and a modified choice (slightly different from geometric series, but reduce the stdv below a threshold). 
![Picked Periods](https://user-images.githubusercontent.com/49893001/95634538-02bb9f80-0a3f-11eb-981f-d2c16084ec94.png)

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
The script askes for several parameters and saves figures and data that are selected by these parameters. The time taken for each pixel is roughly 6 seconds on my pc. Below is an example for the open cluster [FSR2007] 0728. This is a 400 pixel cut, and it took approximately 40 minutes in a test run. 

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
* 10.9.2020: Modified periods to test cnn to avoid big time intervals between phase folded data. This inproves the performance of cnn by reducing continuous same value after interpolation. Also validates a cnn with 500 points is a reasonable limit (limiting stdv of interval/period ~ 0.0006). 

## Contributers

* **Te Han** 
* **Timothy D Brandt** 

## License
 ...
