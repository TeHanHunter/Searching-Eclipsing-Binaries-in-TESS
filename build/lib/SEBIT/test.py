from SEBIT.source import *
from SEBIT.psf import *
from SEBIT.main import *
import numpy as np
import SEBIT
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import pickle
from tqdm.contrib.concurrent import process_map

if __name__ == '__main__':
    global source
    # with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654.pkl', 'wb') as output:
    #     source = Source('NGC 7654', size=30, sector=17)
    #     pickle.dump(source, output, pickle.HIGHEST_PROTOCOL)

    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654.pkl',
              'rb') as input:
        source = pickle.load(input)

    print('Target loaded.')
    source.star_idx(star_idx=None)
    # get 7 nonlinear params
    # source.cguess = psf(source, num=-1)[2:9]
    # epochs = len(source.time)

    """
    # old full frame check 
    def psf_multi(number):
        return psf(source, num=number, c=source.cguess)


    result = process_map(psf_multi, range(len(source.time)))

    base_lc = np.array(result)[:, 1]
    base_std = np.std(base_lc)
    print('Base light curve fitted.')

    star_list = []
    source.gaia_cut = None
    source.z = np.arange(source.size ** 2)
    stdv = []
    for index in range(source.nstars):  # source.nstars
        # star = psf_single(index)
        # star_list.append(star)
        print(index, source.gaia['designation'][index])
        star_idx = [index]
        vary_star = np.where(source.gaia['variability'] == 1)[0]
        star_idx = np.append(star_idx, vary_star)
        source.star_idx(star_idx=star_idx)
        inputs = np.array([[source] * epochs, list(range(epochs)), [source.cguess] * epochs]).transpose()
        with mp.Pool() as pool:
            result = pool.starmap(psf, tqdm(inputs, total=epochs))
        residual = base_lc - np.array(result)[:, 1]
        std = np.std(residual) / base_std
        stdv.append(std)
        print(std, np.median(np.array(result)[:, 2]))
        plt.plot(residual)
        plt.plot(base_lc - np.mean(base_lc))
        plt.show()
        if std > 0.05:
            source.gaia['variability'][index] = 1
            base_lc = np.array(result)[:, 1]
            base_std = np.std(base_lc)
    """


    def variable_check(number):
        source.cut(number)
        index = np.where(source.gaia_cut['variability'] >= 1)[0]
        index_cut = np.append(source.star_index, index)

        source.star_idx(star_idx=index)
        mesh = Mesh(source)  #TODO: check for size of mesh = size of cut
        source.cguess = psf(source, num=-2, mesh=mesh)[2+len(index):9+len(index)]
        result = []
        for i in range(len(source.time)):
            result.append(psf(source, num=i, c=source.cguess, mesh=mesh))
        result_contam = np.array(result)
        contam_std = np.std(result_contam[:, 1])

        source.star_idx(star_idx=index_cut)
        mesh = Mesh(source)
        result = []
        for i in range(len(source.time)):
            result.append(psf(source, num=i, c=source.cguess, mesh=mesh))
        result_float = np.array(result)
        residual = result_contam[:, 1] - result_float[:, 1]
        std = np.std(residual) / contam_std
        print(std)
        if std > 0.5:
            source.gaia['variability'][number] = 1
        return

    # def psf_(numb):
    #     return psf(source=source, num=numb, c=source.cguess, mesh=mesh)
    #
    # def variable_check(number):
    #     source.cut(number)
    #     index = np.where(source.gaia_cut['variability'] >= 1)[0]
    #     index_cut = np.append(source.star_index, index)
    #
    #     source.star_idx(star_idx=index)
    #     mesh = Mesh(source)
    #     source.cguess = psf(source, num=-2, mesh=mesh)[2 + len(index):9 + len(index)]
    #     result_contam = np.array(process_map(psf_, range(len(source.time))))
    #     contam_std = np.std(result_contam[:, 1])
    #
    #     source.star_idx(star_idx=index_cut)
    #     mesh = Mesh(source)
    #     result_float = np.array(process_map(psf_, range(len(source.time))))
    #     residual = result_contam[:, 1] - result_float[:, 1]
    #     std = np.std(residual) / contam_std
    #     print(std)
    #     if std > 0.5:
    #         source.gaia['variability'][number] = 1
    #     return


    # for i in range(100):
    #     variable_check(i)

    # source.star_idx(star_idx=np.where(source.gaia['variability'] == 1)[0])

    # def psf_multi(number):
    #     return psf(source, num=number)
    #
    #
    # result = process_map(psf_multi, range(len(source.time)))
    # with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/result_NGC_7654.pkl', 'wb') as output:
    #     pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)

    # with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/result.pkl', 'rb') as input:
    #     result = pickle.load(input)

    # array([0, 1, 6, <7>, 9, 10, 11, 12, 13, 17, 20, 21, 22, 24, 26, 29, 30,
    #        31, 33, 35, 36, 38, 39, 41, 42, 43, 44, 46, 47])
    # star 7 is correctly recognized as variable, but its feature is not removed enough.
    # The reason is that it is out of the frame.

    # cresult = process_map(psf_multi, range(epochs))

    # empty space: ra 193 dec 27
    # get base light curve

    # inputs = np.array([[source] * epochs, list(range(epochs)), [source.cguess] * epochs]).transpose()
    # with mp.Pool() as pool:
    #     result = pool.starmap(psf, tqdm(inputs, total=epochs))

    # diff = np.sum(np.abs(source.flux - np.mean(source.flux, axis=0)), axis=0)
    # plt.imshow(diff, vmin=np.min(diff), vmax=np.min(diff) + 15000)
    # plt.show()
    # array([-0.10480037601724611, -0.004314153623546809,
    #    -0.0010446719341780499, 0.5419793904971424, -0.007218268580517231,
    #    0.41317655646473495, 2.8016381453659793], dtype=object)
