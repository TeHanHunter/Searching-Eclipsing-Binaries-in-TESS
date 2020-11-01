#!/usr/bin/env python
# coding: utf-8

# # Section 4: CNN on TESS

# In[1]:

import os
import sys
import pickle
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from wotan import flatten
from astropy.wcs import WCS
from astropy.io import fits
from astropy.io import ascii
from sklearn import preprocessing
from astroquery.mast import Tesscut
from progress.bar import ChargingBar
from astroquery.mast import Catalogs
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from astropy.table import Table, Column, MaskedColumn
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Concatenate, Flatten, Dropout
colors = [(1,1,0.5,c) for c in np.linspace(0,1,100)]


target_name = input('Target Identifier: ')
FOV = input('FOV in arcmin (max 33) [Default: 5]: ') or '5'
size = np.int(float(FOV) * 3)
radSearch = size * 21 * 0.707 / 3600  #radius in degrees 
Sample_number = 500
cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
def make_cnn(maxlen):
    
    input_local = Input(shape=(maxlen,1))
    x = Conv1D(16, 5, strides=1)(input_local)
    #x = Conv1D(16, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)
    x = Conv1D(8, 5, strides=1)(x)
    #x = Conv1D(8, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)
    
    xf = Flatten()(x)
    z = Dense(64, activation='relu')(xf)
    #z = Dropout(0.1)(z)
    z = Dense(32, activation='relu')(z)
    z = Dense(8, activation='relu')(z)

    output = Dense(1, activation='sigmoid', name='main_output')(z)
    model = Model(inputs=input_local, outputs=output)
    
    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                optimizer=SGDsolver,
                metrics=['accuracy'])
    return model

tensorflow.get_logger().setLevel('ERROR') ## ignore internal TensorFlow Warning message
cnn = make_cnn(Sample_number)
cnn_weights = input('Trained cnn .h5 file [Default: tess_cnn.h5]: ') or 'tess_cnn.h5'
cnn.load_weights(str(cnn_weights))

catalogData = Catalogs.query_object(target_name, radius = radSearch, catalog = "TIC")
# print(catalogData['ID', 'Tmag', 'Jmag', 'ra', 'dec', 'objType'])

# Create a list of nearby bright stars (tess magnitude less than 14) from the rest of the data for later.
bright = catalogData['Tmag'] < 14

# Make it a list of Ra, Dec pairs of the bright ones. This is now a list of nearby bright stars.
nearbyStars = list( map( lambda x,y:[x,y], catalogData[bright]['ra'], catalogData[bright]['dec'] ) )

#show sectors 
ra = catalogData[0]['ra']
dec = catalogData[0]['dec']
coord = SkyCoord(ra, dec, unit = "deg")
sectorTable = Tesscut.get_sectors(coord)
print('#################################')
print(sectorTable)
print('#################################')

#get FFI cutout
hdulist = Tesscut.get_cutouts(coord, size)
hdu1 = hdulist[0]
firstImage = hdu1[1].data['FLUX'][0]
wcs = WCS(hdu1[2].header)

def aperture_phot(image, aperture):
    """
    Sum-up the pixels that are in the aperture for one image.
    image and aperture are 2D arrays that need to be the same size.
    
    aperture is a boolean array where True means to include the light of those pixels.
    """
    flux = np.sum(image[aperture])

    return flux

def make_lc(flux_data, aperture):
    """
    Apply the 2d aperture array to the and time series of 2D images. 
    Return the photometric series by summing over the pixels that are in the aperture.
    
    Aperture is a boolean array where True means it is in the desired aperture.
    """
    
    flux = np.array(list (map (lambda x: aperture_phot(x, aperture), flux_data) ) )

    return flux



data_time = hdu1[1].data['TIME']
data_flux = hdu1[1].data['FLUX']
data_flux_err = hdu1[1].data['FLUX_ERR']
data_quality = hdu1[1].data['QUALITY']

data_time = data_time[np.where(data_quality == 0)]
data_flux = data_flux[np.where(data_quality == 0),:,:][0]
data_flux_err = data_flux_err[np.where(data_quality == 0),:,:][0]
data_quality = data_quality[np.where(data_quality == 0)]
#Remove background
aperture = hdu1[2].data == 1
flux1 = make_lc(data_flux, aperture)

bkgAperture = data_flux[0] < np.percentile(data_flux[0], 4)
bkgFlux1 = make_lc(data_flux, bkgAperture)

flux_2d = data_flux.reshape(np.shape(data_time)[0],size ** 2)
yy, background, xx= np.meshgrid(data_flux[0,:,0], bkgFlux1, data_flux[0,0,:])
bkgSubFlux = data_flux - background / np.sum(bkgAperture)

Predict_max = np.zeros((size, size))
print('Please choose a threshold to save a figure. This number is the prediction given by cnn, \
indicating the likelihood of including eclipsing binaries in a light curve. 0 is the lowest (keeping every pixel), and 1 is the highest (maybe excluding everything).')
quality = input('Threshold [Default: 0.95]: ') or '0.95'
location = input('Saving figures to [Default: root]: ')
mylist = np.arange(size ** 2)


first_trial = 200
second_trial = 100

###produce periods
a_0 = 0.1
r = 1.023
length = first_trial
geometric = [a_0 * r ** (n - 1) for n in range(1, length + 1)]

a_sam = 0.09
r_sam = 1.000047
leng = int(1e5)
sample_period = np.array([a_sam * r_sam ** (n - 1) for n in range(1, leng + 1)])

std = np.zeros(len(sample_period))
bar = ChargingBar('Sampling Best Trial Periods: ', max = 100, suffix = '%(percent).1f%% Elapsed: %(elapsed)ds Remaining: %(eta)ds')
for i in range(len(sample_period)):
    p = sample_period[i]
    t_pf = data_time%p
    t_pf_sort = np.sort(t_pf)
    gap = np.diff(t_pf_sort)
    std[i] = np.std(gap/p)
    if (i%1000 == 0):
        bar.next()
bar.finish()

std_geo = np.zeros(len(geometric))
for i in range(len(geometric)):
    p = geometric[i]
    t_pf = data_time%p
    t_pf_sort = np.sort(t_pf)
    gap = np.diff(t_pf_sort)
    std_geo[i] = np.std(gap/p)

plt.figure(figsize = (15,8))
plt.plot(sample_period,std, lw = .5, c = 'silver')
plt.plot([0,10],[0.0006,0.0006], c ='C0')
plt.plot([0,10],[0.0005,0.0005], c ='C1')
plt.plot([0,10],[0.0004,0.0004], c ='C2')
plt.xlabel('Period (days)')
plt.ylabel('Stdv of (gap/period)')
#plt.ylim(0, 1.2 * np.max(std_geo))
patch1 = mpatches.Patch(color='silver', label='Sampled Periods')
patch2 = mpatches.Patch(color='C0', label='0.0006')
patch3 = mpatches.Patch(color='C1', label='0.0005')
patch4 = mpatches.Patch(color='C2', label='0.0004')
plt.legend(handles=[patch1, patch2, patch3, patch4], bbox_to_anchor=(0.9, 0.9))
plt.savefig(location  + 'STDV Thereshold choice.png', dpi = 300)

print('Now look at the produced image showing period vs standard deviation of time intervals. Choose a threshold to \
draw test periods from. The data would be better spaced if the threshold is smaller, but notice \
to make sure there are nearby available periods from 0 to 10 days.') 
stdv_threshold = input('Threshold of time interval standard deviation [Default 0.0005]: ') or '0.0005'
    
f = interp1d(sample_period[np.where(std < float(stdv_threshold))],std[np.where(std < float(stdv_threshold))], kind='nearest')
std_mod_p = f(geometric)
mod_p = np.zeros(len(std_mod_p))
for i in range(len(std_mod_p)):    
    mod_p[i] = sample_period[np.where(std == std_mod_p[i])]
ascii.write([mod_p], location + 'modified_geometric.dat', names=['mod_p'], overwrite=True)

mod_periods = np.zeros((first_trial,second_trial))
for i in range(len(mod_p)):
    mod_periods[i] = np.linspace(mod_p[i]/ np.sqrt(r),mod_p[i] * np.sqrt(r), second_trial)

std_mod_periods = f(mod_periods.reshape((1,first_trial *second_trial ))[0])
mod_periods = np.zeros(len(std_mod_periods))
for i in range(len(std_mod_periods)):    
    mod_periods[i] = sample_period[np.where(std == std_mod_periods[i])]
ascii.write(mod_periods.reshape((first_trial,second_trial)).transpose(), location + 'modified_periods.dat', overwrite=True)

plt.figure(figsize = (15,8))
plt.plot(sample_period,std, lw = .2, c = 'silver')
plt.plot(sample_period[np.where(std < float(stdv_threshold))],std[np.where(std < float(stdv_threshold))],'.', c = 'C3', ms = .5)
#plt.plot(geometric, bad_flux/100000 + 0.01, lw = 1, marker = '.', c = 'C2')
plt.plot(mod_periods,std_mod_periods, lw = .5, c = 'C9')
plt.plot(geometric,std_geo,lw = 1, marker = '.',ms = 5, c = 'C1')
plt.plot(mod_p,std_mod_p,lw = 1, marker = '.',ms = 5, c = 'C0')
plt.xlabel('Period (days)')
plt.ylabel('Stdv of (gap/period)')
plt.ylim(0, 1.2 * np.max(std_geo))
patch1 = mpatches.Patch(color='C1', label='Geometric Series')
patch2 = mpatches.Patch(color='C0', label='Modified Geometric (cnn first trial)')
patch3 = mpatches.Patch(color='C9', label='Modified Periods (cnn second trial)')
patch4 = mpatches.Patch(color='silver', label='Sampled Periods')
patch5 = mpatches.Patch(color='C3', label='Sampled Periods with STDV < ' + stdv_threshold)
plt.legend(handles=[patch1, patch2, patch3, patch4, patch5], bbox_to_anchor=(0.9, 0.9))
plt.savefig(location  + 'Picked Periods.png', dpi = 300)

file = open(location + target_name + '.txt', 'w') 
file.write('Target Identifier:' + target_name + '\n' +
           'FOV in arcmin (max 33) [Default: 5]:' + FOV + '\n' +
           'Trained cnn .h5 file [Default: tess_cnn_weights.h5]:' + cnn_weights + '\n' +
           'Threshold [Default: 0.95]:' + quality + '\n' +
           'Saving figures to [Default: root]:' + location + '\n' +
           'Threshold of time interval standard deviation [Default 0.0005]: ' + stdv_threshold)
file.close()

###Start CNN
bar = ChargingBar('Finished pixels: ', max = len(mylist), suffix = '%(percent).1f%% Elapsed: %(elapsed)ds Remaining: %(eta)ds')
for l in range(np.shape(data_flux)[2]):
    for i in range(np.shape(data_flux)[1]):
        flux_raw = bkgSubFlux[:,i,l]
        flux_raw -= np.min(flux_raw)
        #eliminate outliers
        mean = np.mean(flux_raw)
        #standard_deviation = np.std(flux_raw)
        #distance_from_mean = abs(flux_raw - mean)
        #max_deviations = 3
        #not_outlier = distance_from_mean < max_deviations * standard_deviation
        flux_1d = flatten(data_time, flux_raw, break_tolerance = 0.5 , window_length = 2, edge_cutoff = 0.5, return_trend = False)

        #remove nan in flux (causes trouble for cnn)
        index = np.where(flux_1d >= 0)
        flux_1d = flux_1d[index]
        time_1d = data_time[index]
        flux_err_1d = data_flux_err[:,i,l][index]
        quality_1d = data_quality[index]
        
        #make CNN tests
        period = mod_p       
        t_0 = np.linspace(-0.1,0.1,5)
        predict = np.zeros((len(period),len(t_0)))
        argsort = flux_1d.argsort()
        
        for j in range(len(period)):
            p = period[j]
            t_zero = np.mean(time_1d[argsort][0:5]% p)/ p
            for k in range(len(t_0)):
                t_pf = np.array((time_1d + (0.5 - t_zero + t_0[k]) * p) % p )
                t = np.linspace(np.min(t_pf), np.max(t_pf), Sample_number)
                f = interp1d(t_pf, flux_1d, kind='nearest')
                flux = f(t)
                #np.max(flux) - np.min(flux) np.percentile(flux, 100) - np.percentile(flux, 0)
                flux /= (np.max(flux) - np.min(flux)) / 4
                flux -= np.average(flux)
                predict[j][k] = cnn.predict(flux.reshape((1, Sample_number, 1)))
        idx = np.where(predict == np.max(predict))
         
        ### repeat in the region near the best result of first step for higher precision
        period_ = mod_periods.reshape((first_trial,second_trial))[np.where(period == period[idx[0][0]])][0]
        predict = np.zeros((len(period_),len(t_0)))

    
        for j in range(len(period_)):
            p = period_[j]
            t_zero = np.mean(time_1d[argsort][0:5]% p) / p
            for k in range(len(t_0)):
                t_pf = np.array((time_1d + (0.5 - t_zero + t_0[k]) * p) % p )
                t = np.linspace(np.min(t_pf), np.max(t_pf), Sample_number)
                f = interp1d(t_pf, flux_1d, kind='nearest')
                flux = f(t)
                flux /= (np.max(flux) - np.min(flux)) / 4
                flux -= np.average(flux)
                predict[j][k] = cnn.predict(flux.reshape((1, Sample_number, 1)))
                
        idx = np.where(predict == np.max(predict))
        p = period_[idx[0][0]]
        t_zero = np.mean(time_1d[argsort][0:5]% p) / p
        t_pf = np.array((time_1d + (0.5 - t_zero + t_0[idx[1][0]]) * p) % p )
        ###
        t = np.linspace(np.min(t_pf), np.max(t_pf), Sample_number)
        inter = interp1d(t_pf, flux_1d, kind='nearest')
        cnn_flux = inter(t)
        #cnn_flux /= (np.max(cnn_flux) - np.min(cnn_flux)) / 4
        #cnn_flux -= np.average(cnn_flux)
        ###
        
        Predict_max[i,l] = np.max(predict)
        #plot
        if np.max(predict) > float(quality):
            fig = plt.figure(constrained_layout = False, figsize=(15, 7))
            gs = fig.add_gridspec(3, 3)
            gs.update(wspace = 0.3, hspace = 0.4)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1:])
            ax3 = fig.add_subplot(gs[1:, 0], projection = wcs)
            ax4 = fig.add_subplot(gs[1:, 1:])
                
            ax1.axis('off')
            ax1.text(0.1,0.8,'%.4f' % np.max(predict), fontsize=40)
            ax1.text(0.1,0.5,'CNN prediction: %.8f' % np.max(predict), fontsize=10)
            ax1.text(0.1,0.3,'Period: %.8f' % period_[idx[0][0]], fontsize=10)
            #ax1.text(0.1,0.1,'Other bests: ' + str(0.5 * len(idx) - 1), fontsize=10)
            ax1.text(0.1,0.1,'RA, Dec: ' + str(wcs.all_pix2world([[l,i]],0)), fontsize=10)
            ax2.plot(data_time, flux_raw, color = 'silver')
            ax2.set_title(target_name + ' x = ' + str(l ) + ', y = ' + str(i ), fontsize = 15)
            ax2.set_ylabel('Background Subtracted Flux')
            ax2.set_xlabel('Time (TBJD)')
            ax2.plot(data_time[index], flux_raw[index], ms = 2, marker = '.', c = 'C1', linestyle = '')
            ax3.imshow(firstImage, origin = 'lower', cmap = plt.cm.YlGnBu_r, vmax = np.percentile(firstImage, 98),
                       vmin = np.percentile(firstImage, 5))
            ax3.grid(axis = 'both',color = 'white', ls = 'solid')
            nearbyLoc = wcs.all_world2pix(nearbyStars[0:],0)
            ax3.imshow(bkgAperture,cmap=cmap)
            ax3.scatter(nearbyLoc[0:, 0], nearbyLoc[0:, 1], s = 200 / size, color = 'C1')
            ax3.set_xlim(-0.5,size - 0.5)
            ax3.set_ylim(-0.5,size - 0.5)
            ax3.set_xlabel('RA', fontsize = 12)
            ax3.set_ylabel('Dec', fontsize = 12)
            ax3.scatter(l,i, marker = 's', s = 50000 / size ** 2, facecolors='none', edgecolors='r')
            ax4.errorbar(t_pf,flux_1d * mean, flux_err_1d, marker = '.', ms = 2, ls = '', elinewidth = 0.4, ecolor = 'silver', c = 'darkgrey')
            ax4.plot(t,cnn_flux * mean, c = 'C1', lw = 0.8)
            ax4.set_xlabel('Period = %.4f' % period_[idx[0][0]])
            ax4.set_ylabel('Detrended Phase Folded Flux')
            plt.savefig(location  + '[' + str(l) + ','+ str(i) + '] %.4f' %np.max(predict) + '.png', dpi = 100)
            data = Table([time_1d, flux_1d * mean, flux_err_1d, quality_1d], names=['TBJD', 'bkgsubflux', 'flux_err', 'quality'])
            ascii.write(data, location + 'TESS_' + str(target_name) + '[' + str(l) + ','+ str(i)+ '].dat', overwrite=True)
        else:
            pass
        bar.next()
bar.finish()

fig = plt.figure(constrained_layout = False, figsize=(8, 7))
b = plt.imshow(Predict_max, origin = 'lower', cmap = 'bone', vmax = 1,vmin = np.percentile(Predict_max, 50))
c = plt.axes([0.9, 0.12, 0.02, 0.75])
cbar = plt.colorbar(b, cax = c)
cbar.ax.tick_params(labelsize = 10)
plt.savefig(location  + 'Prediction_colormap.png', dpi = 100)
