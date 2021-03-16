from SEBIT.source import *
from SEBIT.psf import *
from SEBIT.cnn import *
from SEBIT.visual import *


def search(name, size=15):
    size = int(size)
    if type(name) != str:
        raise TypeError('Target identifier or coordinates of string type.')
    if type(size) != int:
        raise TypeError('Pixel size of FFI cut must be an integer.')
    target = Source(name, size=size)

    # First fit --> get nonlinear parameters of moffat
    target.threshold('none')
    psf_result = psf(-1,target)

    return target

if __name__ == '__main__':
    print('Testing on NGC 7654')
    target = Source('NGC 7654', size=15)
    target.threshold('none')
    r = psf(0, target)
