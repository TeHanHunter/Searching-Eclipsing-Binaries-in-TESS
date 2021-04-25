import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle


def bilinear(x, y, repeat=25):
    """
    value = a + (b - a) * y + (a - c) * x + (b + d - a - c) * x * y
    coefficients of a, b, c, d [1 + x - y - x * y, y + x * y, -x - x * y, x * y]
    (x+1)*(1-y)

    a, c = array[0]
    b, d = array[1]
    """
    return np.array([1 + x - y - x * y, -x - x * y, y + x * y, x * y] * repeat)


def get_psf(source):
    psf_size = 11
    half_size = int((psf_size - 1) / 2)
    over_size = psf_size * 2 + 1
    # global flux_ratio, x_shift, y_shift, x_round, y_round, x_sign, y_sign
    # nstars = source.nstars
    size = source.size  # TODO: must be even?
    flux_ratio = np.array(source.gaia['tess_flux_ratio'])
    x_shift = np.array(source.gaia[f'Sector_{source.sector}_x'])
    y_shift = np.array(source.gaia[f'Sector_{source.sector}_y'])
    x_round = np.round(x_shift).astype(int)
    y_round = np.round(y_shift).astype(int)
    x_sign = np.sign(x_shift - x_round)
    y_sign = np.sign(y_shift - y_round)

    left = np.maximum(0, x_round - half_size)
    right = np.minimum(size, x_round + half_size) + 1
    down = np.maximum(0, y_round - half_size)
    up = np.minimum(size, y_round + half_size) + 1
    x_residual = x_shift % 0.5
    y_residual = y_shift % 0.5
    # pixel
    x_p = np.arange(size)
    y_p = np.arange(size)
    coord = np.arange(size ** 2).reshape(size, size)
    # try 7*7
    A = np.zeros((size ** 2, over_size ** 2 + 1))
    A[:, -1] = np.ones(size ** 2)
    for i in range(len(flux_ratio)):
        x_psf = 2 * x_p[left[i]:right[i]] - 2 * x_round[i] + psf_size - 0.5 - 0.5 * x_sign[i]
        y_psf = 2 * y_p[down[i]:up[i]] - 2 * y_round[i] + psf_size - 0.5 - 0.5 * y_sign[i]
        x_psf, y_psf = np.meshgrid(x_psf, y_psf)  # super slow here
        a = np.array(x_psf + y_psf * over_size, dtype=np.int64)
        a = a.flatten()
        index = coord[down[i]:up[i], left[i]:right[i]]
        A[np.repeat(index, 4), np.array([a, a + 1, a + over_size, a + over_size + 1]).flatten(order='F')] += \
            flux_ratio[i] * bilinear(x_residual[i], y_residual[i], repeat=len(a))

    scaler = np.sqrt(source.flux_err[0].flatten() ** 2 + source.flux[0].flatten())
    fit = np.linalg.lstsq(A / scaler[:, np.newaxis], source.flux[0].flatten() / scaler, rcond=None)[0]
    # fit = np.linalg.lstsq(A, source.flux[0].flatten(), rcond=None)[0]
    fluxfit = np.dot(A, fit)
    # np.lstsq(A/err[np.newaxis, :], b/err)
    # err = np.sqrt(bg**2 + cts)
    return fit, fluxfit


if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_90,0.pkl',
              'rb') as input: source = pickle.load(input)

    fit, fluxfit = get_psf(source)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plot0 = ax[0].imshow(source.flux[0], vmin=0, vmax=np.max(source.flux[0]), origin='lower')
    plot1 = ax[1].imshow(fluxfit.reshape(source.size, source.size), vmin=0, vmax=np.max(source.flux[0]), origin='lower')
    plot2 = ax[2].imshow(
        (source.flux[0] - fluxfit.reshape(source.size, source.size)).reshape((source.size, source.size)),origin='lower')
    ax[0].set_title('Raw Data')
    ax[1].set_title('Mesh_linear Model')
    ax[2].set_title('Residual')
    fig.colorbar(plot0, ax=ax[0])
    fig.colorbar(plot1, ax=ax[1])
    fig.colorbar(plot2, ax=ax[2])
    plt.show()
    plt.imshow(fit[0:-1].reshape(23, 23), origin='lower')
    plt.show()




"""
# for one star, one pixel
size = 20  # TODO: must be even?
# unique for star
flux_ratio = np.array([1])
x_shift = np.array([5.3])
y_shift = np.array([4.6])
x_round = np.round(x_shift)
y_round = np.round(y_shift)
x_sign = np.sign(x_shift - x_round)
y_sign = np.sign(y_shift - y_round)

A = np.zeros((size ** 2, 121))  # all pixels * all x_i
for i in range(size ** 2):
    if x_round[0] - 2 <= x_p[i % size] <= x_round[0] + 2 and y_round[0] - 2 <= y_p[i // size] <= y_round[0] + 2:
        # coordinate of pixel [x_p - x_round + 2, y_p - y_round + 2]
        # coordinate of lower left psf [x_psf, y_psf]
        x_psf = 2 * x_p[i % size] - 2 * x_round[0] + 4.5 - 0.5 * x_sign[0]
        y_psf = 2 * y_p[i // size] - 2 * y_round[0] + 4.5 - 0.5 * y_sign[0]
        a = int(x_psf + y_psf * 11)
        index = [a, a + 1, a + 11, a + 12]
        A[i][np.array(index)] = flux_ratio[0] * bilinear(x, y)
"""

# plot interpolation
# x, y = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.imshow(fit[0:-1].reshape(23, 23), extent=[-5.75, 5.75, -5.75, 5.75], origin='lower')
# x_shift = np.array(source.gaia[f'Sector_{source.sector}_x'])[7]
# y_shift = np.array(source.gaia[f'Sector_{source.sector}_y'])[7]
# # ax.imshow(psf, extent=[-5.5 + x_shift, 5.5 + x_shift,
# #                        -5.5 + y_shift, 5.5 + y_shift],
# #           alpha=1,origin='lower', cmap='bone')
# ax.scatter(x + x_shift % 1 - 0.5, y + y_shift % 1 - 0.5, marker='s', facecolor='None', edgecolors='w', s=480)
# ax.plot(x + x_shift % 1 - 0.5, y + y_shift % 1 - 0.5, '.w', ms=3)
# plt.show()

# # Initializing value of x-axis and y-axis
# # in the range -1 to 1
# size = 20  # TODO: must be even?
# over_sample_size = size * 2 + 1
# x_shift = 15.3
# y_shift = 4.5
# # Intializing sigma and muu
# sigma = 1
# muu = 0.
#
# x_round = round(x_shift)
# y_round = round(y_shift)
#
# flux_cube = np.zeros((1, size, size))
#
# left = max(0, x_round - 2)
# right = min(size, x_round + 2) + 1
# down = max(0, y_round - 2)
# up = min(size, y_round + 2) + 1
#
# # Calculating Gaussian array
# x_psf, y_psf = np.meshgrid(np.linspace(-2.5, 2.5, 11), np.linspace(-2.5, 2.5, 11))
# dst = np.sqrt(x_psf * x_psf + y_psf * y_psf)
# gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
# interp = interpolate.RectBivariateSpline(np.linspace(-2.5, 2.5, 11), np.linspace(-2.5, 2.5, 11), gauss)
# x, y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
# psf = interp.ev(x - x_shift + x_round, y - y_shift + y_round)
#
# flux_cube[0, down:up, left:right] = psf[max(0, 2 - x_round):min(5, size - x_round + 3),
#                                     max(0, 2 - y_round):min(5, size - y_round + 3)]
#
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.imshow(flux_cube[0], vmin=0, vmax=1, origin='lower', cmap='gray')
# ax.scatter(x_shift, y_shift)
# plt.show()

# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.imshow(gauss, extent=[-2.75, 2.75, -2.75, 2.75], vmin=0, vmax=1, origin='lower', cmap='gray')
# ax.imshow(psf, extent=[-2.5 + x_shift, 2.5 + x_shift, -2.5 + y_shift, 2.5 + y_shift], vmin=-0.1, vmax=1, alpha=1,
#           origin='lower', cmap='bone')
# ax.scatter(x + x_shift, y + y_shift, marker='s', facecolor='None', edgecolors='w', s=2200)
# ax.plot(x + x_shift, y + y_shift, '.w', ms=3)
# ax.scatter(x_psf, y_psf, s=np.log(gauss + 1.01) * 30, c='r')
# plt.show()

# def blockshaped(arr, nrows, ncols):
#     """
#     Return an array of shape (n, nrows, ncols) where
#     n * nrows * ncols = arr.size
#     If arr is a 2D array, the returned array should look like n subblocks with
#     each subblock preserving the "physical" layout of arr.
#     """
#     h, w = arr.shape
#     assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
#     assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
#     return arr.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)
#
#
# def unblockshaped(arr, h, w):
#     """
#     Return an array of shape (h, w) where
#     h * w = arr.size
#
#     If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
#     then the returned array preserves the "physical" layout of the sublocks.
#     """
#     n, nrows, ncols = arr.shape
#     return arr.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)
