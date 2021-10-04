import pickle
import numpy as np
import matplotlib.pyplot as plt
from wotan import flatten

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    #
    # eb_list = np.loadtxt('/mnt/c/users/tehan/desktop/eb_candidate_new.csv', delimiter=',')
    lightcurve = np.load('/mnt/c/users/tehan/desktop/lightcurves.npy')
    #
    # for i in range(int(eb_list[-1, -1]) + 1):
    #     index = np.where(eb_list[:, -1] == i)[0]
    #     fig = plt.figure(constrained_layout=False, figsize=(10, 2 * len(index) + 1))
    #     gs = fig.add_gridspec(len(index), 5)
    #     gs.update(wspace=0.5, hspace=0.4)
    #     for j in range(len(index)):
    #         star_index = int(eb_list[index[j]][0])
    #         flatten_lc = flatten(source.time, lightcurve[star_index], break_tolerance=0.1, window_length=1, return_trend=False)
    #         ax1 = fig.add_subplot(gs[j, 0:4])
    #         ax2 = fig.add_subplot(gs[j, 4])
    #         ax1.plot(source.time, flatten_lc / np.median(flatten_lc), '.k', ms=5)
    #         ax2.plot(source.time % eb_list[index[j]][1], flatten_lc / np.median(flatten_lc), '.k', ms=1)
    #         ax1.set_title(f'Star {star_index}, p = {eb_list[index[j]][1]}')
    #     plt.savefig(f'/mnt/c/users/tehan/desktop/eb_candidate_new/{i}.png', dpi=300)
    #     plt.show()
