from SEBIT.source import *
from SEBIT.psf import *
# from SEBIT.cnn import *


# from SEBIT.visual import *


def search(name: str, size=15):
    size = int(size)
    if type(size) != int:
        raise TypeError('Pixel size of FFI cut must be an integer.')
    target = Source(name, size=size)

    # First fit --> get nonlinear parameters of moffat
    target.threshold('none')
    psf_result = psf(-1, target)
    c = psf_result.nonlinparam

    # Second fit --> use nonlinear params to fit each star
    target.threshold('all')
    result = []
    for i in range(len(target.time)):
        result.append(psf(i, target, c=c))

    return result


if __name__ == '__main__':
    print('Testing on NGC 7654')
    r = search('NGC 7654', size=5)
    print(r)
