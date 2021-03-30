from SEBIT.source import *
from SEBIT.psf import *
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
import tqdm

source = None


def psf_multi(number):
    return psf(source, num=number)


def search(name: str, size=15, sector=None, search_gaia=True, threshold=15):
    global source
    size = int(size)
    if type(size) != int:
        raise TypeError('Pixel size of FFI cut must be an integer.')
    source = Source(name, size=size, sector=sector, search_gaia=search_gaia)
    print('Target downloaded.')
    source.threshold(mag_threshold=threshold)
    source.cguess = psf(source)[2:8]
    print(source.cguess)
    result = process_map(psf_multi, range(len(source.time)))
    print(np.array(result).transpose()[2:8])
    print(np.array(result).transpose()[2:8][0])
    source.threshold(star_idx=1, mag_threshold=threshold)

    c_result = np.array(result)[:, 2:8]
    star_0 = []
    for i in range(len(source.time)):
        star_0.append(psf(source, num=i, c=c_result[i]))
    star_0 = np.array(star_0).transpose()
    return star_0
    # TODO: first search change to all frames; c update; maybe multiprocessing


if __name__ == '__main__':
    print('Testing on NGC 7654')
    r = search('351.4069010 61.6466715', sector=18, threshold=16)
