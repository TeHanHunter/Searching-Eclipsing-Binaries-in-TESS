from SEBIT.source import *
from SEBIT.psf import *
from tqdm.contrib.concurrent import process_map

source = None
c_result = None


def psf_multi(number):
    return psf(source, num=number)


def psf_single(number):
    source.cut(number)
    index = source.star_index
    index.extend(np.where(source.gaia_cut['variability'] >= 1)[0])
    print(index)
    source.star_idx(star_idx=index)
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
    source.star_idx(star_idx=[0])
    source.cguess = psf(source)[3:10]
    result = process_map(psf_multi, range(len(source.time)))
    c_result = np.array(result)[:, 3:10]
    c_result[:, 6:10] = np.meshgrid(np.median(np.array(c_result)[:, 6:10], axis=0), np.zeros(len(c_result)))[0]
    star_list = []
    for i, index in enumerate(source.inner_star):
        star = psf_single(index)
        star_list.append(star)
        if np.std(star.flux) >= 0.01:
            source.gaia['variability'][index] = 1

    # star_list = process_map(psf_single, source.inner_star)
    # TODO: filter out near edge targets Done
    return star_list, source


if __name__ == '__main__':
    print('Testing on NGC 7654')
    r = search('351.4069010 61.6466715', sector=18, threshold=15)
