import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import ascii
from wotan import flatten

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    lightcurve = np.load('/mnt/c/users/tehan/desktop/lightcurves.npy')


    def flatten_lc(source, lightcurve, index, bg_mod=0):
        flatten_lc = flatten(source.time, (lightcurve[index] - bg_mod) / (np.median(lightcurve[index]) - bg_mod),
                             window_length=1,
                             method='biweight',
                             return_trend=False)
        return flatten_lc


    index = [699, 77, 1251, 469, 1585]
    period = [1.01968, 1.9221, 2.2895, 6.126, 6.558]
    lc = np.zeros((len(index), len(source.time)))
    for i in range(len(index)):
        lc[i] = flatten_lc(source, lightcurve, index[i])
        if i == 2:
            print(lightcurve[index[i]][0] - np.median(source.flux[:, 10, 24] * source.gaia['tess_flux_ratio'][index[i]]))
            lc[i] = flatten_lc(source, lightcurve, index[i], bg_mod=
            lightcurve[index[i]][0] - np.median(source.flux[:, 10, 24] * source.gaia['tess_flux_ratio'][index[i]]))

    fig = plt.figure(constrained_layout=False, figsize=(10, 2 * len(index) + 1))
    gs = fig.add_gridspec(len(index), 4)
    gs.update(wspace=0.5, hspace=0.6)
    t1 = 530
    t2 = 555
    t3 = 1080
    for i in range(len(index)):
        ax1 = fig.add_subplot(gs[i, 0:2])
        ax2 = fig.add_subplot(gs[i, 2:])
        ax1.plot(source.time, lc[i], '.k', ms=1, zorder=0)
        ax1.scatter(source.time[t3:], lc[i][t3:], marker='x', c='r', s=7, linewidths=0.5)
        ax1.scatter(source.time[t1:t2], lc[i][t1:t2], marker='x', c='r', s=7, linewidths=0.5, label='TESS outliers')
        # ax1.plot(source.time[t2:t2 + 500], lc[i][t2:t2 + 500], '.', c='C0', ms=3)
        ax2.plot(source.time[0:t1] % period[i], lc[i][0:t1], '.k', ms=1)
        ax2.plot(source.time[t2:t3] % period[i], lc[i][t2:t3], '.k', ms=1, label='TESS')
        ylim = ax2.get_ylim()
        ax2.set_ylim((ylim[0] - 0.02, ylim[1] + 0.02))
        ylim = ax2.get_ylim()
        try:
            data = ascii.read(f'/mnt/c/users/tehan/desktop/eb_candidate_new/ZTF/{index[i]}_g.csv')
            data.remove_rows(np.where(data['catflags'] != 0))
            tbjd = data['hjd'] - 2457000
            mag = data['mag']
            flux = 10 ** (- mag / 2.5)  # 3.208e-10 *
            ax2_ = ax2.twinx()
            ax2_.plot(tbjd % period[i], flux / np.median(flux), 'x', color='green', ms=3, label='ZTF g-band')
            # ax2_.set_ylabel('ZTF mag')
            ax2_.set_ylim(ylim)
            ax2_.get_yaxis().set_visible(False)
            # ax2_.tick_params(axis='y', colors='k')
        except:
            pass
        try:
            data = ascii.read(f'/mnt/c/users/tehan/desktop/eb_candidate_new/ZTF/{index[i]}_r.csv')
            data.remove_rows(np.where(data['catflags'] != 0))
            tbjd = data['hjd'] - 2457000
            mag = data['mag']
            flux = 10 ** ((4.74 - mag) / 2.5)
            ax2__ = ax2.twinx()
            ax2__.scatter(tbjd % period[i], flux / np.median(flux), facecolors='none', edgecolors='orangered', s=5,
                          label='ZTF r-band')
            ax2__.set_ylim(ylim)
            ax2__.get_yaxis().set_visible(False)
            if i == 2:
                ax2.set_ylim([0.65, 1.1])
                ax2_.set_ylim([0.65, 1.1])
                ax2__.set_ylim([0.65, 1.1])
        except:
            pass
        ax1.set_xlabel('TBJD', labelpad=0)
        ax1.set_ylabel('Normalized Flux', labelpad=0)
        ax2.set_xlabel('Phase (days)', labelpad=0)
        ax2.set_ylabel('Normalized Flux', labelpad=0)
        ax1.set_title(f'{source.gaia[index[i]]["designation"]}')
        ax2.set_title(f'P = {period[i]}' + f' TESS magnitude = {source.gaia[index[i]]["tess_mag"]:.2f}')
    ax1.legend(bbox_to_anchor=(0.9, -.35))
    ax2.legend(bbox_to_anchor=(-.8, -.35))
    ax2_.legend(bbox_to_anchor=(0.5, -.35))
    ax2__.legend(bbox_to_anchor=(.9, -.35))
    # plt.savefig(f'/mnt/c/users/tehan/desktop/eb_candidate_new/EBs.png', dpi=300)
    plt.show()
