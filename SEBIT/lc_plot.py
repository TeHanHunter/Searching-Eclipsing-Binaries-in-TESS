import numpy as np
import matplotlib.pyplot as plt
import pickle
from wotan import flatten

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    period = 1.01972534
    time = np.load('/mnt/c/users/tehan/desktop/eleanor_time.npy')
    eleanor_aperture = np.load('/mnt/c/users/tehan/desktop/eleanor_aperture.npy')
    eleanor_PSF = np.load('/mnt/c/users/tehan/desktop/eleanor_PSF.npy')
    moffat = np.load('/mnt/c/users/tehan/desktop/moffat.npy')
    epsf = np.load('/mnt/c/users/tehan/desktop/epsf_lc.npy')
    flatten_lc, trend_lc = flatten(source.time, epsf / np.median(epsf), window_length=1, method='biweight',
                                   return_trend=True)
    flatten_lc_, trend_lc_ = flatten(source.time, moffat / np.median(moffat), window_length=1, method='biweight',
                                   return_trend=True)
    fig = plt.figure(constrained_layout=False, figsize=(15, 10))
    gs = fig.add_gridspec(4, 5)
    gs.update(wspace=0.3, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0:4])
    ax2 = fig.add_subplot(gs[1, 0:4])
    ax3 = fig.add_subplot(gs[2, 0:4])
    ax4 = fig.add_subplot(gs[3, 0:4])
    ax5 = fig.add_subplot(gs[0, 4])
    ax6 = fig.add_subplot(gs[1, 4])
    ax7 = fig.add_subplot(gs[2, 4])
    ax8 = fig.add_subplot(gs[3, 4])

    ax1.plot(time, eleanor_aperture / np.median(eleanor_aperture), '.k', ms=5)
    ax2.plot(time, eleanor_PSF / np.median(eleanor_PSF), '.k', ms=5)
    # ax3.plot(source.time, moffat / np.median(moffat), '.k', ms=5)
    ax3.plot(source.time, flatten_lc_, '.k', ms=5)
    ax4.plot(source.time, flatten_lc, '.k', ms=5)
    # ax4.plot(source.time, (flatten_lc * np.median(epsf) + 42) / np.median(epsf + 42), '.k', ms=5)
    ax5.plot(time % period, eleanor_aperture / np.median(eleanor_aperture), '.k', ms=1)
    ax6.plot(time % period, eleanor_PSF / np.median(eleanor_PSF), '.k', ms=1)
    # ax7.plot(source.time % period, moffat / np.median(moffat), '.k', ms=1)
    ax7.plot(source.time % period, flatten_lc_, '.k', ms=1)
    ax8.plot(source.time % period, flatten_lc, '.k', ms=1)
    # ax8.plot(source.time % period, (flatten_lc * np.median(epsf) + 42) / np.median(epsf + 42), '.k', ms=1)
    ax1.set_title('eleanor aperture')
    ax2.set_title('eleanor PSF')
    ax3.set_title('detrended Moffat PSF')
    ax4.set_title('detrended ePSF')
    plt.savefig('/mnt/c/users/tehan/desktop/light_curve_comparison_both_detrended.png', dpi=300)
    plt.show()
