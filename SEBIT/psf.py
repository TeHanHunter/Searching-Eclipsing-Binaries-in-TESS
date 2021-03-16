import numpy as np
from scipy import optimize


class Linmodel:
    def __init__(self):
        self.par = None
        self.y = None
        self.cov = None
        self.cov_inv = None


class PsfResult(object):
    def __init__(self):
        self.linparam = None
        self.nonlinparam = None
        self.time = None
        self.result = None


def chisq_model(par, model, flux, source):
    return np.sum((model(par, flux, source).y - flux) ** 2)


def moffat(x_, y_, a, b, c, beta, size):
    x, y = np.meshgrid(np.linspace(0, size - 1, size), np.linspace(0, size - 1, size))
    x -= x_
    y -= y_
    g = (1 + (a * x ** 2 + 2 * b * x * y + c * y ** 2)) ** (- beta)
    return g


def contamination(lin_pars, c, source):
    flux = np.zeros((source.size, source.size))
    for i in range(source.nstars):
        if i in source.star_idx:
            continue
        else:
            flux += lin_pars[1] * np.array(source.gaia['tess_flux_ratio'])[i] * moffat(
                np.array(source.gaia['Sector_{}_x'.format(source.sector)])[i] + c[0],
                np.array(source.gaia['Sector_{}_y'.format(source.sector)])[i] + c[1], c[2], c[3], c[4],
                c[5], source.size).reshape((source.size, source.size))
    return flux + lin_pars[0] * np.ones((source.size, source.size))


def moffat_model(c, flux, source):  # size, nstars, idx, flux_ratio, x_shift, y_shift
    # x = z // source.size
    # y = z % source.size
    A = np.ones((len(source.z), 2 + len(source.star_idx)))
    A[:, 0] = 1  # F_bg
    flux_conta = np.zeros(source.size ** 2)
    for i in range(source.nstars):
        if i in source.star_idx:
            continue
        else:
            flux_conta += np.array(source.gaia['tess_flux_ratio'])[i] * moffat(
                np.array(source.gaia['Sector_{}_x'.format(source.sector)])[i] + c[0],
                np.array(source.gaia['Sector_{}_y'.format(source.sector)])[i] + c[1], c[2], c[3], c[4],
                c[5], source.size).reshape(source.size ** 2)
    A[:, 1] = flux_conta  # A
    for j, index in enumerate(source.star_idx):
        A[:, j + 2] = moffat(np.array(source.gaia['Sector_{}_x'.format(source.sector)])[index] + c[0],
                             np.array(source.gaia['Sector_{}_y'.format(source.sector)])[index] + c[1],
                             c[2], c[3], c[4], c[5], source.size).reshape(source.size ** 2)  # F_ebs

    result = Linmodel()
    if np.isnan(np.sum(A)):
        result.y = np.inf
        return result
    fluxfit = np.zeros(len(source.z))
    Ap = A
    bp = flux
    fit = np.linalg.lstsq(Ap, bp, rcond=None)[0]

    fluxfit += np.dot(A, fit)

    result.y = fluxfit
    result.par = fit
    return result


def psf(num, source, cfit=None):
    if num == -1:
        flux = (np.sum(source.flux, axis=0) / len(source.flux)).reshape(source.size ** 2)
        cfit = cfit
    else:
        flux = source.flux[num].reshape(source.size ** 2)
        cfit = optimize.minimize(chisq_model, source.cguess, (moffat_model, flux, source), method="Powell", bounds=source.var_to_bounds).x
    c_result = moffat_model(cfit, flux, source)
    aperture = source.flux[num] - contamination(c_result.par, cfit, source)
    r = PsfResult()
    r.linparam = list(c_result.par)
    r.nonlinparam = list(cfit)
    r.time = source.time[num]
    r.result = aperture
    return r
