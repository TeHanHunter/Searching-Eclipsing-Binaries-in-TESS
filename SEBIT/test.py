from SEBIT.source import *
from SEBIT.psf import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    source = Source('351.4069010 61.6466715', sector=18)
    source.threshold(star_idx=1)
    mesh = Mesh(source)
    x = mesh.x - mesh.x_shift
    y = mesh.y - mesh.y_shift
    flux_cube = (1 + (2 * x ** 2 + 2 * 0.1 * x * y + 2 * y ** 2)) ** (- 2)

    # plt.imshow(np.sum(flux_cube, axis=0), origin='lower')
    # plt.show()
    A = mesh.A
    A[:, 0] = 1  # F_bg
    A[:, 1] = np.sum(
        (flux_cube * mesh.flux_ratio)[np.array(list(set(np.arange(source.nstars)) ^ set(source.star_idx)))],
        axis=0).reshape(source.size ** 2)  # F_norm
    for j, index in enumerate(source.star_idx):
        A[:, j + 2] = flux_cube[index].reshape(source.size ** 2)  # F_ebs