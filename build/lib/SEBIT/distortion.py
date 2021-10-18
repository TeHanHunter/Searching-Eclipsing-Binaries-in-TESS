from SEBIT.source import *
from SEBIT.psf import *
import numpy as np
import matplotlib.pyplot as plt
import pickle


def paraboloid(z_0, z_1, z_2, z_3, z_4, z_5, scale=0.1):
    """
    z = c_1 x^2 + c_2 y^2 + c_3 xy + c_4 x + c_5 y + c_6

    Parameters:
        z_0: x = 0, y = 0
        z_1: x = 0.1, y = 0
        z_2: x = 0, y = 0.1
        z_3: x = -0.1, y = 0
        z_4: x = 0, y = -0.1
        z_5: x = 0.1, y = 0.1
        scale:
    """
    # c_1 = 50 * (z_3 + z_1 - 2 * z_0)
    # c_2 = 50 * (z_4 + z_2 - 2 * z_0)
    # c_4 = 5 * (z_1 - z_3)
    # c_5 = 5 * (z_2 - z_4)
    # c_6 = z_0
    # c_3 = 100 * z_5 - 100 * c_6 - 10 * c_4 - 10 * c_5 - c_1 - c_2
    # x_max = (2 * c_1 * c_4 - c_3 * c_5) / (c_3 ** 2 - 4 * c_1 * c_2)
    # y_max = (2 * c_1 * c_5 - c_3 * c_4) / (c_3 ** 2 - 4 * c_1 * c_2)
    # return x_max, y_max

    c_1 = (z_3 + z_1 - 2 * z_0) / (2 * scale ** 2)
    c_2 = (z_4 + z_2 - 2 * z_0) / (2 * scale ** 2)
    c_4 = (z_1 - z_3) / (2 * scale)
    c_5 = (z_2 - z_4) / (2 * scale)
    c_6 = z_0
    c_3 = z_5 * scale ** -2 - 100 * c_6 * scale ** -2 - c_4 * scale ** -1 - c_5 * scale ** -1 - c_1 - c_2
    x_max = (2 * c_1 * c_4 - c_3 * c_5) / (c_3 ** 2 - 4 * c_1 * c_2)
    y_max = (2 * c_1 * c_5 - c_3 * c_4) / (c_3 ** 2 - 4 * c_1 * c_2)
    return x_max, y_max


def residual(left, right, down, up, residual, star_num=1000):
    z = np.zeros(star_num)
    for j in range(star_num):
        z[j] = np.sqrt(np.sum(np.square(residual[down[j]:up[j], left[j]:right[j]])))
    return z


def change_c(c, position=0, scale=0.1):
    if position == 0:
        pass
    elif position == 1:
        c[0] = scale
    elif position == 2:
        c[1] = scale
    elif position == 3:
        c[0] = -scale
    elif position == 4:
        c[1] = -scale
    elif position == 5:
        c[0] = scale
        c[1] = scale
    return c

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_50.pkl',
              'rb') as input:
        source = pickle.load(input)
    x_shift = np.array(source.gaia[f'Sector_{source.sector}_x'])
    y_shift = np.array(source.gaia[f'Sector_{source.sector}_y'])
    x_round = np.round(x_shift).astype(int)
    y_round = np.round(y_shift).astype(int)
    left = np.maximum(0, x_round - 2)
    right = np.minimum(source.size, x_round + 2) + 1
    down = np.maximum(0, y_round - 2)
    up = np.minimum(source.size, y_round + 2) + 1

    source.star_idx(star_idx=None)
    mesh = Mesh(source)
    result = psf(source, mesh=mesh)
    c = result[2:9]
    star_num = 5000
    z = np.zeros((6, star_num))
    for i in range(6):
        c_ = change_c(c, position=i)
        result = psf(source, mesh=mesh, c=c_, aperture=True)
        z[i, :] = residual(left, right, down, up, result[-1], star_num=star_num)
        print(c)
    x_max, y_max = paraboloid(z[0], z[1], z[2], z[3], z[4], z[5])

    plt.figure(figsize=(8, 8))
    for i in range(star_num):
        plt.arrow(x_shift[i], y_shift[i], 1000 * x_max[i], 1000 * y_max[i], length_includes_head=True, width=0.05)
    plt.savefig('/mnt/c/users/tehan/desktop/flux_.jpg', dpi=500)
    plt.show()
