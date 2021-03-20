from SEBIT.source import *
from SEBIT.psf import *


# from SEBIT.cnn import *


# from SEBIT.visual import *


def search(name: str, size=15, threshold=5):
    size = int(size)
    if type(size) != int:
        raise TypeError('Pixel size of FFI cut must be an integer.')
    target = Source(name, size=size)

    # First fit --> get nonlinear parameters of moffat
    target.threshold(mag_diff=threshold)
    psf_result = psf(target, num=-1)
    c = psf_result.nonlinparam

    # Second fit --> use nonlinear params to fit each star
    target.threshold(star_idx='all', mag_diff=threshold)
    # TODO: more options
    result = []
    for i in range(len(target.time)):
        result.append(psf(target, num=i, c=c))

    return result


if __name__ == '__main__':
    print('Testing on NGC 7654')
    r = search('NGC 7654', size=5)
    print(r)
