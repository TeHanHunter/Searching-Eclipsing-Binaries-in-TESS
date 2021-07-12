import numpy as np
import matplotlib.pyplot as plt
from SEBIT.source import *
from SEBIT.psf import *
import pickle
from tqdm.contrib.concurrent import process_map


def psf_multi(number):
    return psf(source, num=number)


if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)

    source.star_idx(star_idx=77)
    mesh = Mesh(source)
    # result = psf(source, mesh=mesh, aperture=True)
    # c_result = np.array(result)[3:10]
    c_result = [-0.1336967657243149, 0.009780820798694904, -7.69258082838413e-05, 1.0266558030680182,
                0.004231685426634509, 0.7145283185015417, 2.0238829196982184]
    light_curve = np.zeros(len(source.time))
    # bkg = np.zeros(len(source.time))
    for i in range(len(source.time)):
        result = psf(source, num=i, c=c_result, mesh=mesh, aperture=True)
        # bkg[i] = result[0]
        light_curve[i] = np.sum(result[-1][40, 64])
        print(i)

