from SEBIT.source import *
from SEBIT.psf import *
# from SEBIT.cnn import *
# from SEBIT.visual import *


def search(name: str, size=15, threshold=5):
    size = int(size)
    if type(size) != int:
        raise TypeError('Pixel size of FFI cut must be an integer.')
    source = Source(name, size=size)
    print('Target downloaded.')
    # First fit --> get nonlinear parameters of moffat
    source.threshold(mag_diff=threshold)
    psf_result = psf(source, num=-1)
    c = psf_result.nonlinparam
    print('Moffat nonlinear parameters fitted.')
    # Second fit --> use nonlinear params to fit each star
    source.threshold(star_idx='all', mag_diff=threshold)
    result = []
    for i in range(len(source.time)):
        result.append(psf(source, num=i, c=c))
    result = np.array(result).transpose()
    return result
    # TODO: first search change to all frames; c update; maybe multiprocessing

if __name__ == '__main__':
    print('Testing on NGC 7654')
    r = search('NGC 7654', size=5)
    print(r)
