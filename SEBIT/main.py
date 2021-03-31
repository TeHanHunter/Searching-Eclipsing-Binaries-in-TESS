from SEBIT.source import *
from SEBIT.psf import *
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
import tqdm

source = None
c_result = None


def psf_multi(number):
    return psf(source, num=number)


def psf_single(number):
    source.cut(number)
    result = []
    for i in range(len(source.time)):
        result.append(psf(source, num=i, c=c_result[i]))
    result = np.array(result)
    return Star(number, source, result)


def search(name: str, size=15, sector=None, search_gaia=True, threshold=15):
    global source, c_result
    size = int(size)
    if type(size) != int:
        raise TypeError('Pixel size of FFI cut must be an integer.')
    source = Source(name, size=size, sector=sector, search_gaia=search_gaia, mag_threshold=threshold)
    print('Target downloaded.')
    source.star_idx(star_idx=None)
    source.cguess = psf(source)[2:8]
    result = process_map(psf_multi, range(len(source.time)))
    c_result = np.array(result)[:, 2:8]

    star_list = process_map(psf_single, range(source.nstars))
    return star_list


if __name__ == '__main__':
    print('Testing on NGC 7654')
    r = search('351.4069010 61.6466715', sector=18, threshold=15)
