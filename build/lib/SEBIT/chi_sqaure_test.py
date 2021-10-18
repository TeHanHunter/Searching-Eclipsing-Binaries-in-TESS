import numpy as np
from scipy import optimize
from SEBIT.ePSF import *
import pickle

if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    factor = 4
    epsf = np.load('/mnt/c/users/tehan/desktop/epsf_.npy')
    A, star_info, over_size, x_round, y_round = get_psf(source, factor=factor)
    # original chi_square
    # chi_square = np.zeros(len(source.time))
    # for i in range(len(source.time)):
    #     fluxfit = np.dot(A[0:source.size ** 2], epsf[i])
    #     chi_2 = np.sum((source.flux[i].flatten() - fluxfit) ** 2)
    #     chi_square[i] = chi_2
    #     print(i, chi_2)
    # np.save('/mnt/c/users/tehan/desktop/chi_2/chi_2', chi_square)
    # stars = [4505, 5667, 9588, 8457, 3969, 3406, 699, 872, 1055, 717]
    # stars = [77, 4756, 6346, 6831, 8327, 954, 5818, 5454, 818, 2858, 6674, 7876, 1488, 1623]
    stars = [1585, 1671, 5281, 9686]
    for j in range(len(stars)):
        star_num = stars[j]
        r_A = reduced_A(A, star_info, star_num=star_num)
        # float chi_square
        chi_square_ = np.zeros(len(source.time))
        coeff = np.zeros(len(source.time))
        for i in range(len(source.time)):
            other_flux = np.dot(r_A[0:source.size ** 2], epsf[i])
            target_flux = np.dot((A - r_A)[0:source.size ** 2], epsf[i])
            B = np.stack((np.ones(source.size ** 2), other_flux, target_flux)).T
            variable_fit = np.linalg.lstsq(B, source.flux[i].flatten())[0]
            coeff[i] = variable_fit[2]
            fluxfit = np.dot(B, variable_fit)
            chi_2 = np.sum((source.flux[i].flatten() - fluxfit[0:source.size ** 2]) ** 2)
            chi_square_[i] = chi_2
            print(i)
            np.save(f'/mnt/c/users/tehan/desktop/chi_2/chi_2_{star_num}', chi_square_)

    # chi_square = np.zeros((len(stars), len(source.time)))
    # for j in range(len(stars)):
    #     star_num = stars[j]
    #     chi_square[j] = np.load(f'/mnt/c/users/tehan/desktop/chi_2/chi_2_{star_num}.npy')
    #
    # plt.plot(np.average(chi_square.T, axis=0))
    # plt.show()
