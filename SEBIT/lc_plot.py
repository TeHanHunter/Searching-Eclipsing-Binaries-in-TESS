import numpy as np
import matplotlib.pyplot as plt
import pickle
from wotan import flatten

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    period = 2.2903
    t1_ = 435
    t2_ = 455
    t3_ = 940

    t1 = 530
    t2 = 555
    t3 = 1080

    time = np.load('/mnt/c/users/tehan/desktop/eleanor_time.npy')
    eleanor_aperture = np.load('/mnt/c/users/tehan/desktop/eleanor_aperture_cross_1251.npy')
    eleanor_PSF = np.load('/mnt/c/users/tehan/desktop/eleanor_PSF_1251.npy')
    moffat = np.load('/mnt/c/users/tehan/desktop/moffat_1251.npy')
    lightcurve = np.load('/mnt/c/users/tehan/desktop/lightcurves.npy')

    bg_mod = lightcurve[1251][0] - np.median(source.flux[:, 10, 24] * source.gaia['tess_flux_ratio'][1251])
    # epsf
    flatten_lc, trend_lc = flatten(source.time, (lightcurve[1251] - bg_mod) / np.median((lightcurve[1251] - bg_mod)),
                                   window_length=1,
                                   method='biweight',
                                   return_trend=True)
    # moffat
    flatten_lc_, trend_lc_ = flatten(source.time, (moffat - bg_mod) / np.median(moffat - bg_mod), window_length=1,
                                     method='biweight',
                                     return_trend=True)
    # eleanor gaussian
    flatten_lc__, trend_lc__ = flatten(time, eleanor_PSF / np.median(eleanor_PSF), window_length=1, method='biweight',
                                       return_trend=True)
    fig = plt.figure(constrained_layout=False, figsize=(15, 10))
    gs = fig.add_gridspec(4, 5)
    gs.update(wspace=0.3, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[1, 0:3])
    ax3 = fig.add_subplot(gs[2, 0:3])
    ax4 = fig.add_subplot(gs[3, 0:3])
    ax5 = fig.add_subplot(gs[0, 3:5])
    ax6 = fig.add_subplot(gs[1, 3:5])
    ax7 = fig.add_subplot(gs[2, 3:5])
    ax8 = fig.add_subplot(gs[3, 3:5])

    ax1.plot(time, eleanor_aperture / np.median(eleanor_aperture), '.k', ms=5)
    ax2.plot(time, flatten_lc__, '.k', ms=5)
    ax3.plot(source.time, flatten_lc_, '.k', ms=5)
    ax4.plot(source.time, flatten_lc, '.k', ms=5)

    ax5.plot(time[0:t1_] % period, eleanor_aperture[0:t1_] / np.median(eleanor_aperture), '.k', ms=1)
    ax5.plot(time[t2_:t3_] % period, eleanor_aperture[t2_:t3_] / np.median(eleanor_aperture), '.k', ms=1)
    ax6.plot(time[0:t1_] % period, flatten_lc__[0:t1_], '.k', ms=1)
    ax6.plot(time[t2_:t3_] % period, flatten_lc__[t2_:t3_], '.k', ms=1)
    ax7.plot(source.time[0:t1] % period, flatten_lc_[0:t1], '.k', ms=1)
    ax7.plot(source.time[t2:t3] % period, flatten_lc_[t2:t3], '.k', ms=1)
    ax8.plot(source.time[0:t1] % period, flatten_lc[0:t1], '.k', ms=1)
    ax8.plot(source.time[t2:t3] % period, flatten_lc[t2:t3], '.k', ms=1)

    ax1.set_title('eleanor aperture')
    ax2.set_title('detrended eleanor Gaussian PSF')
    ax3.set_title('detrended Moffat PSF')
    ax4.set_title('detrended ePSF')

    ax1.set_ylim(0.993, 1.006)
    ax5.set_ylim(0.993, 1.006)
    ax2.set_ylim(0.993, 1.006)
    ax6.set_ylim(0.993, 1.006)
    ax3.set_ylim(0.65, 1.15)
    ax7.set_ylim(0.65, 1.15)
    ax4.set_ylim(0.65, 1.15)
    ax8.set_ylim(0.65, 1.15)

    ax1.plot(time[t3_:], eleanor_aperture[t3_:] / np.median(eleanor_aperture), '.', c='silver', ms=5)
    ax1.plot(time[t1_:t2_], eleanor_aperture[t1_:t2_] / np.median(eleanor_aperture), '.', c='silver', ms=5)
    ax2.plot(time[t3_:], flatten_lc__[t3_:], '.', c='silver', ms=5)
    ax2.plot(time[t1_:t2_], flatten_lc__[t1_:t2_], '.', c='silver', ms=5)
    ax3.plot(source.time[t3:], flatten_lc_[t3:], '.', c='silver', ms=5)
    ax3.plot(source.time[t1:t2], flatten_lc_[t1:t2], '.', c='silver', ms=5)
    ax4.plot(source.time[t3:], flatten_lc[t3:], '.', c='silver', ms=5)
    ax4.plot(source.time[t1:t2], flatten_lc[t1:t2], '.', c='silver', ms=5)

    plt.savefig('/mnt/c/users/tehan/desktop/light_curve_comparison_1251.png', dpi=300)
    plt.show()

    # time_arg = np.argsort(time % period)
    # moffat_time_arg = np.argsort(source.time % period)
