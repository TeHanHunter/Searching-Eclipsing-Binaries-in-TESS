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
from tqdm.contrib.concurrent import process_map
from astropy.table import Table, Column, MaskedColumn, hstack
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Concatenate, Flatten, Dropout

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Source(object):
    """
    Get FFI cut using TESScut

    Parameters
    ----------
    name : str or float
        Target identifier (e.g. "NGC 7654" or "M31"),
        or coordinate in the format of ra dec (e.g. 351.40691 61.646657)
    size : int, optional
        The side length in pixel  of TESScut image
    sector : int, optional
        The sector for which data should be returned. If None, returns the first observed sector
    search_gaia : boolean, optional
        Whether to search gaia targets in the field

    """

    def __init__(self, name, size=15, sector=None, search_gaia=True):
        super(Source, self).__init__()
        self.name = name
        self.size = size
        catalog = Catalogs.query_object(self.name, radius=self.size * 21 * 0.707 / 3600, catalog="TIC")
        ra = catalog[0]['ra']
        dec = catalog[0]['dec']
        coord = SkyCoord(ra, dec, unit="deg")
        hdulist = Tesscut.get_cutouts(coord, self.size)
        sector_table = Tesscut.get_sectors(coord)
        if sector is None:
            self.sector = sector_table['sector'][0]
            hdu = hdulist[0]
        else:
            # TODO: test sector number
            self.sector = sector
            hdu = hdulist[list(sector_table['sector']).index(sector)]
        wcs = WCS(hdu[2].header)
        data_time = hdu[1].data['TIME']
        data_flux = hdu[1].data['FLUX']
        data_flux_err = hdu[1].data['FLUX_ERR']
        data_quality = hdu[1].data['QUALITY']

        data_time = data_time[np.where(data_quality == 0)]
        data_flux = data_flux[np.where(data_quality == 0), :, :][0]
        data_flux_err = data_flux_err[np.where(data_quality == 0), :, :][0]
        self.wcs = wcs
        self.time = data_time
        self.flux = data_flux
        self.flux_err = data_flux_err
        if search_gaia:
            catalogdata = Catalogs.query_object(self.name, radius=self.size * 21 * 0.707 / 3600, catalog="Gaia")
            catalogdata.sort("phot_g_mean_mag")
            gaia_targets = catalogdata[
                'designation', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ra', 'dec']
            x = np.zeros(len(gaia_targets))
            y = np.zeros(len(gaia_targets))
            for j, designation in enumerate(gaia_targets['designation']):
                pixel = self.wcs.all_world2pix(
                    np.array([gaia_targets['ra'][j], gaia_targets['dec'][j]]).reshape((1, 2)), 0)
                x[j] = pixel[0][0]
                y[j] = pixel[0][1]
            t = Table()
            t[f'Sector_{self.sector}_x'] = x
            t[f'Sector_{self.sector}_y'] = y
            gaia_targets = hstack([gaia_targets, t])
            self.gaia = gaia_targets
        else:
            self.gaia = None


if __name__ == '__main__':
    target = Source(name='NGC 7654', size=99)
