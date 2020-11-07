# Searching-Eclipsing-Binaries-with-TESS
Using the TESS 30-min interval full frame images, this code searches for eclipsing binaries with convolutional neural network (CNN). With `Batman`, the code produces pseudo light curves of EBs and trains a CNN model with them. This model is then used on each pixel of full frame images produced by TESScut. The output includes all light curves with high value cnn predictions above an arbitrary threshold. The image below is a sample result. 

<!-- <img src=https://user-images.githubusercontent.com/49893001/97091618-0a0fab00-15f2-11eb-926e-097c558bb119.png width = '1024'> -->
![10,60  1 0000](https://user-images.githubusercontent.com/49893001/97091618-0a0fab00-15f2-11eb-926e-097c558bb119.png)
The periods to test cnn on is carefully chosen to avoid huge time intervals between adjacent data points, which make interpolation do a poor job. The below image shows the difference of standard deviation of time intervals between a blunt choice (geometric series) and a modified choice (slightly different from geometric series, but reduce the stdv below a threshold).

<p align="center">
  <img src=https://user-images.githubusercontent.com/49893001/95634538-02bb9f80-0a3f-11eb-981f-d2c16084ec94.png width = '768'>
</p>

![5,5  0 9974](https://user-images.githubusercontent.com/49893001/98451299-4dd9d880-20f9-11eb-91ad-68c571af6473.png)

TIC 199688409 was observed in 13 sectors and have data spanning nearly a year. The code finds a period with much higher precision (and takes more time). 
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
The script askes for several parameters and saves figures and data that are selected by these parameters. The time taken for each pixel is roughly 6 seconds on my pc. Below is an example for the open cluster Trumpler 5. This is a 5184 pixel cut, and it took approximately 8.5 hours in a test run. 

```
Target Identifier: Trumpler 5
FOV in arcmin (max 33) [Default: 5]: 24
Trained cnn .h5 file [Default: tess_cnn.h5]:
#################################
  sectorName   sector camera ccd
-------------- ------ ------ ---
tess-s0006-1-3      6      1   3
#################################
Please choose a threshold to save a figure. This number is the prediction given by cnn, indicating the likelihood of including eclipsing binaries in a light curve. 0 is the lowest (keeping every pixel), and 1 is the highest (maybe excluding everything).
Threshold [Default: 0.95]: 0.99
Saving figures to [Default: root]: /mnt/c/users/tehan/desktop/Trumpler 10.23/
Sampling Best Trial Periods:  ████████████████████████████████ 100.0% Elapsed: 62s Remaining: 0s
Now look at the produced image showing period vs standard deviation of time intervals. Choose a threshold to draw test periods from. The data would be better spaced if the threshold is smaller, but notice to make sure there are nearby available periods from 0 to 10 days.
Threshold of time interval standard deviation [Default 0.0006]: 0.00045
Finished pixels:  ████████████████████████████████ 100.0% Elapsed: 31102s Remaining: 0s
```
Change the target and size of the cut to test it on any target. Note: Each TESS pixel is about 21 arcsec wide.

## Log
* 8.24.2020: Created Rrepository, uploaded CNN training and Searching EBs folder.
* 8.29.2020: Created .py file to be run in the command line.
* 9.15.2020: Added detrending function (Wotan).
* 9.26.2020: Updated cnn model (tess_cnn_weights.h5) by using real light curves of TESS.
* 10.9.2020: Modified periods to test cnn to avoid big time intervals between phase folded data. This inproves the performance of cnn by reducing continuous same value after interpolation. Also validates a cnn with 500 points is a reasonable limit (limiting stdv of interval/period ~ 0.0006). 
* 10.24.2020: Removed low quality data and added flux error to the output file to be ready for an MCMC fit.
* 10.31.2020: Optimized choice of periods, increasing the speed by a factor of 3. Added colormap of maximum predictions. Added two new cnn models, 'tess_cnn_sparse.h5' and 'tess_cnn_strict.h5'.
* 11.7.2020: Allowed the use of multiple sector data if the target is observed more than once. Tested on targets observed for 13 sectors (nearly a year long).

## Contributers

* **Te Han** 
* **Timothy D Brandt** 

## License
 ...
