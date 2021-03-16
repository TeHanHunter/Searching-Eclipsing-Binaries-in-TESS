from SEBIT.source import *
from SEBIT.psf import *

if __name__ == '__main__':
    target = Source('NGC 7654', size=5)
    target.threshold('none')
    r = psf(0, target)
