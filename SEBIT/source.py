import os
import sys
import warnings
import numpy as np
from astropy.wcs import WCS
from astroquery.mast import Tesscut
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack

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
    z : list
        Parametrized x (z // size) and y (z % size)
    sector : int, optional
        The sector for which data should be returned. If None, returns the first observed sector
    search_gaia : boolean, optional
        Whether to search gaia targets in the field

    Attributes
    ----------
    wcs : astropy.wcs.WCS class
        World Coordinate Systems information of the FFI
    time : numpy.ndarray (1d)
        Time of each frame
    flux : numpy.ndarray (3d)
        Fluxes of each frame, spanning time space
    flux_err : numpy.ndarray (3d)
        Flux errors of each frame, spanning time space
    gaia : astropy.table.table.Table class
        Gaia information including ra, dec, brightness, projection on TESS FFI, etc.
    """
    # variable parameters
    nstars = None
    star_idx = [0]
    cguess = [0, 0, 1, 0, 1, 2]
    var_to_bounds = [(-0.5, 0.5), (-0.5, 0.5), (0, 10.0), (-0.5, 0.5), (0, 10.0), (1, np.inf)]

    def __init__(self, name, size=15, sector=None, search_gaia=True):
        super(Source, self).__init__()
        self.name = name
        self.size = size
        self.z = np.arange(self.size ** 2)
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

            tess_mag = np.zeros(len(gaia_targets))
            for i, designation in enumerate(gaia_targets['designation']):
                dif = gaia_targets['phot_bp_mean_mag'][i] - gaia_targets['phot_rp_mean_mag'][i]
                tess_mag[i] = gaia_targets['phot_g_mean_mag'][
                                  i] - 0.00522555 * dif ** 3 + 0.0891337 * dif ** 2 - 0.633923 * dif + 0.0324473
                if np.isnan(tess_mag[i]):
                    tess_mag[i] = gaia_targets['phot_g_mean_mag'][i] - 0.430
            tess_flux = 10 ** (- tess_mag / 2.5)
            t = Table()
            t[f'tess_mag'] = tess_mag
            t[f'tess_flux'] = tess_flux
            t[f'tess_flux_ratio'] = tess_flux / np.max(tess_flux)
            t[f'Secpsftor_{self.sector}_x'] = x
            t[f'Sector_{self.sector}_y'] = y
            gaia_targets = hstack([gaia_targets, t])
            gaia_targets.sort('tess_mag')
            self.gaia = gaia_targets
        else:
            self.gaia = None

    def threshold(self, star_idx=None, mag_diff=5):
        # TODO: None
        """
        Choose stars of interest (primarily for PSF fitting

        Attributes/
        ----------
        nstars : int
            Number of stars of interest, cut by a magnitude threshold
        star_idx : list or str
            Star indexes for PSF fitting, list of indexes, int, None, or 'all'
        mag_diff : int or float
            Brightness threshold for stars to fit
        """
        nstars = np.where(self.gaia['phot_g_mean_mag'] < (min(self.gaia['phot_g_mean_mag']) + mag_diff))[0][-1]
        self.nstars = nstars
        if star_idx is None:
            self.star_idx = np.array([], dtype=int)
        elif star_idx == 'all':
            self.star_idx = np.arange(self.nstars)
        elif type(star_idx) == int:
            self.star_idx = np.array([star_idx])
        elif type(star_idx) == list and all(isinstance(n, int) for n in star_idx):
            self.star_idx = np.array(star_idx)
        else:
            raise TypeError("Star index (star_idx) type should be a list of ints, int, None or 'all'. ")


if __name__ == '__main__':
    target = Source('NGC 7654')
