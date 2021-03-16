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

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

colors = [(1, 1, 0.5, c) for c in np.linspace(0, 1, 100)]


def make_cnn(maxlen):
    input_local = Input(shape=(maxlen, 1))
    x = Conv1D(16, 5, strides=1)(input_local)
    # x = Conv1D(16, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)
    x = Conv1D(8, 5, strides=1)(x)
    # x = Conv1D(8, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)

    xf = Flatten()(x)
    z = Dense(64, activation='relu')(xf)
    # z = Dropout(0.1)(z)
    z = Dense(32, activation='relu')(z)
    z = Dense(8, activation='relu')(z)

    output = Dense(1, activation='sigmoid', name='main_output')(z)
    model = Model(inputs=input_local, outputs=output)

    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=SGDsolver,
                  metrics=['accuracy'])
    return model


def period(source):
    first_trial = 200
    second_trial = 200
    period_range = 10  # days

    # produce periods
    a_0 = 0.1
    r = (period_range * 10) ** (1 / first_trial)
    length = first_trial
    geometric = [a_0 * r ** (n - 1) for n in range(1, length + 1)]

    a_sam = 0.09
    r_sam = (period_range * 11.2) ** 1e-5
    leng = int(1e5)
    sample_period = np.array([a_sam * r_sam ** (n - 1) for n in range(1, leng + 1)])

    std = np.zeros(len(sample_period))
    for i in range(len(sample_period)):
        p = sample_period[i]
        t_pf = source.time % p
        t_pf_sort = np.sort(t_pf)
        gap = np.diff(t_pf_sort)
        std[i] = np.std(gap / p)

    std_geo = np.zeros(len(geometric))
    for i in range(len(geometric)):
        p = geometric[i]
        t_pf = source.time % p
        t_pf_sort = np.sort(t_pf)
        gap = np.diff(t_pf_sort)
        std_geo[i] = np.std(gap / p)

    stdv_threshold = input('Threshold of time interval standard deviation [Default 0.0005]: ') or '0.0005'

    # first trial periods
    f = interp1d(sample_period[np.where(std < float(stdv_threshold))], std[np.where(std < float(stdv_threshold))],
                 kind='nearest')
    std_mod_p = f(geometric)
    mod_p = np.zeros(len(std_mod_p))
    for i in range(len(std_mod_p)):
        mod_p[i] = sample_period[np.where(std == std_mod_p[i])]
    # ascii.write([mod_p], location + 'modified_geometric.csv', names=['mod_p'], overwrite=True)

    # second trial periods
    mod_periods = np.zeros((first_trial, second_trial))
    for i in range(len(mod_p)):
        mod_periods[i] = np.linspace(mod_p[i] / np.sqrt(r), mod_p[i] * np.sqrt(r), second_trial)
    std_mod_periods = f(mod_periods.reshape((1, first_trial * second_trial))[0])
    mod_periods = np.zeros(len(std_mod_periods))
    for i in range(len(std_mod_periods)):
        mod_periods[i] = sample_period[np.where(std == std_mod_periods[i])]
    # np.savetxt(location + 'modified_periods.csv', mod_periods.reshape((first_trial, second_trial)).transpose(),
    #            fmt='%.6e', delimiter=",")
    return mod_p, mod_periods.reshape((first_trial, second_trial))
