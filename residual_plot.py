import numpy as np
import matplotlib.pyplot as plt
import pickle
from SEBIT.psf import *

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)

    source.star_idx(star_idx=None)
    mesh = Mesh(source)
    c_result = [-0.1336967657243149, 0.009780820798694904, -7.69258082838413e-05, 1.0266558030680182,
                0.004231685426634509, 0.7145283185015417, 2.0238829196982184]
    result = psf(source, c=c_result, mesh=mesh, aperture=True)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plot0 = ax[0].imshow(np.log10(source.flux[0]), vmin=np.min(np.log10(source.flux[0])),
                         vmax=np.max(np.log10(source.flux[0])), origin='lower')
    plot1 = ax[1].imshow(np.log10(source.flux[0] - result[-1]), vmin=np.min(np.log10(source.flux[0])),
                         vmax=np.max(np.log10(source.flux[0])), origin='lower')
    plot2 = ax[2].imshow(result[-1], origin='lower', vmin=- np.max(result[-1]), vmax=np.max(result[-1]), cmap='RdBu')
    ax[0].set_title('Raw Data')
    ax[1].set_title('Moffat Model')
    ax[2].set_title('Residual')
    fig.colorbar(plot0, ax=ax[0])
    fig.colorbar(plot1, ax=ax[1])
    fig.colorbar(plot2, ax=ax[2])
    plt.savefig('/mnt/c/users/tehan/desktop/moffat_residual.png', dpi=300)
    # plt.show()
