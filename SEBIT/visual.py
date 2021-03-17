import os
import sys
import warnings
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from wotan import flatten
from astropy.wcs import WCS
from astropy.io import ascii
from multiprocessing import Pool, Array
from astroquery.mast import Tesscut
from progress.bar import ChargingBar
from astroquery.mast import Catalogs
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from astropy.table import Table, Column, MaskedColumn, hstack
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Concatenate, Flatten, Dropout

# period stdv
plt.figure(figsize=(15, 8))
plt.plot(sample_period, std, lw=.5, c='silver')
plt.plot([0, period_range], [0.0006, 0.0006], c='C0')
plt.plot([0, period_range], [0.0005, 0.0005], c='C1')
plt.plot([0, period_range], [0.0004, 0.0004], c='C2')
plt.xlabel('Period (days)')
plt.ylabel('Stdv of (gap/period)')
# plt.ylim(0, 1.2 * np.max(std_geo))
patch1 = mpatches.Patch(color='silver', label='Sampled Periods')
patch2 = mpatches.Patch(color='C0', label='0.0006')
patch3 = mpatches.Patch(color='C1', label='0.0005')
patch4 = mpatches.Patch(color='C2', label='0.0004')
plt.legend(handles=[patch1, patch2, patch3, patch4], bbox_to_anchor=(0.9, 0.9))
plt.savefig(location + 'STDV Thereshold choice.png', dpi=300)