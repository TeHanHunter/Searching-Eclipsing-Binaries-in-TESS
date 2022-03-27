import numpy as np
import matplotlib.pyplot as plt
from SEBIT.source import *
from SEBIT.psf import *
import pickle
from tqdm import trange
from tqdm.contrib.concurrent import process_map

if __name__ == '__main__':
    with open('/home/tehan/data/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)

    source.star_idx(star_idx=1251)
    mesh = Mesh(source)
    # result = psf(source, num=-1, mesh=mesh, aperture=True)
    # c_result = np.array(result)[3:10]
    # c_result = [-0.1336967657243149, 0.009780820798694904, -7.69258082838413e-05, 1.0266558030680182,
    #             0.004231685426634509, 0.7145283185015417, 2.0238829196982184]
    # array([-0.08941984059470387, -0.08816619846011438, 0.0004021380928119409,
    #        0.49016294268811206, -9.99084615880041e-05, 0.3829768565632826,
    #        2.855838047023347], dtype=object)
    light_curve = np.zeros(len(source.time))
    bkg = np.zeros(len(source.time))
    for i in trange(len(source.time)):
        result = psf(source, num=i, mesh=mesh, aperture=True)
        light_curve[i] = np.sum(result[-1][32, 47])
    np.save('/home/tehan/data/Searching-Eclipsing-Binaries-in-TESS/moffat_1251.npy', light_curve)
