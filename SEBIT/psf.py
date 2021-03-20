import numpy as np
from scipy import optimize


class Linmodel:
    """
    Linear model with parameters

    Attributes
    ----------
    par : list
        List of parameters
    y : list
        List of function output
    """

    def __init__(self):
        self.par = None
        self.y = None


class PsfResult(object):
    """
    Saving results of PSF

    Attributes
    ----------
    linparam : list
        List of linear parameters
    nonlinparam : list
        List of nonlinear parameters
    time : float
        Time of this frame
    result : numpy.ndarray (2d)
        Array of contamination removed FFI
    """

    def __init__(self):
        self.linparam = None
        self.nonlinparam = None
        self.time = None
        self.result = None


def chisq_model(par, model, flux, source):
    """
    Chi-square model for linear minimization

    Parameters
    ----------
    par : list
        List of parameters
    model : fn
        Function of the model used (moffat)
    flux : numpy.ndarray (2d)
        Fluxes of this frame
    source : SEBIT.source.Source class
        Source class object with data
    """
    return np.sum((model(par, flux, source).y - flux) ** 2)


def moffat(x_, y_, a, b, c, beta, size):
    """
    Moffat model of one star at certain location of the frame

    Parameters
    ----------
    x_ : float
        x pixel location of the star
    y_ : float
        y pixel location of the star
    a : float
        Coefficient of x^2 term
    b : float
        Coefficient of 2xy term
    c : float
        Coefficient of y^2 term
    beta : float
        Power of moffat model (positive, negative sign is left in function)
    """
    x, y = np.meshgrid(np.linspace(0, size - 1, size), np.linspace(0, size - 1, size))
    x -= x_
    y -= y_
    g = (1 + (a * x ** 2 + 2 * b * x * y + c * y ** 2)) ** (- beta)
    return g
    # TODO: vectorize x, y and g
    # TODO: set a moffat boundary?


def contamination(lin_pars, c, source):
    """
    Calculate contamination from other sources

    Parameters
    ----------
    lin_pars : list
        List of linear parameters
    c : list
        List of nonlinear parameters
    source : SEBIT.source.Source class
        Source class object with data
    """
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


def moffat_model(c, flux, source):
    """
    Moffat models summed with linear parameters fitted

    Parameters
    ----------
    c : list
        List of nonlinear parameters
    flux : numpy.ndarray (2d)
        Fluxes of this frame
    source : SEBIT.source.Source class
        Source class object with data
    """
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


def psf(source, num=0, c=None):
    """
    PSF model

    Parameters
    ----------
    num : int
        Number of frame that is fitted
    source : SEBIT.source.Source class
        Source class object with data
    c : list
        List of nonlinear parameters,
        if None: fit for nonlinear parameters
        if list: use given nonlinear parameters
    """
    if num == -1:
        flux = (np.sum(source.flux, axis=0) / len(source.flux)).reshape(source.size ** 2)
    else:
        flux = source.flux[num].reshape(source.size ** 2)
    if c is None:
        cfit = optimize.minimize(chisq_model, source.cguess, (moffat_model, flux, source), method="Powell",
                                 bounds=source.var_to_bounds, options={'disp': True}).x
        # TODO: method?
    else:
        cfit = c
    c_result = moffat_model(cfit, flux, source)
    aperture = source.flux[num] - contamination(c_result.par, cfit, source)
    r = PsfResult()
    r.linparam = list(c_result.par)
    r.nonlinparam = list(cfit)
    r.time = source.time[num]
    r.result = aperture
    return r
