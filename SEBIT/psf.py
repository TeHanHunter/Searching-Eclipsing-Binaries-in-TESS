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


def moffat(x_, y_, a, b, c, beta, source):
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
    x, y, flux_ratio = np.meshgrid(np.arange(source.size), np.arange(source.size),
                                   source.gaia['tess_flux_ratio'][0:source.nstars])
    x_shift = np.meshgrid(np.zeros(source.size), np.zeros(source.size),
                          np.array(source.gaia['Sector_{}_x'.format(source.sector)])[0:source.nstars])[2]
    y_shift = np.meshgrid(np.zeros(source.size), np.zeros(source.size),
                          np.array(source.gaia['Sector_{}_y'.format(source.sector)])[0:source.nstars])[2]
    x = x - np.add(x_shift, x_)
    y = y - np.add(y_shift, y_)
    x = x.transpose(2, 0, 1)
    y = y.transpose(2, 0, 1)
    flux_ratio = flux_ratio.transpose(2, 0, 1)
    flux_cube = (1 + (a * x ** 2 + 2 * b * x * y + c * y ** 2)) ** (- beta)
    return flux_cube, flux_ratio
    # TODO: set a moffat boundary?


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
    flux_cube, flux_ratio = moffat(c[0], c[1], c[2], c[3], c[4], c[5], source)
    A = np.ones((len(source.z), 2 + len(source.star_idx)))
    A[:, 0] = 1  # F_bg
    A[:, 1] = np.sum((flux_cube * flux_ratio)[np.array(list(set(np.arange(source.nstars)) ^ set(source.star_idx)))],
                     axis=0).reshape(source.size ** 2)  # F_norm

    for j, index in enumerate(source.star_idx):
        A[:, j + 2] = flux_cube[j].reshape(source.size ** 2)  # F_ebs

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
    flux_cube, flux_ratio = moffat(cfit[0], cfit[1], cfit[2], cfit[3], cfit[4], cfit[5], source)
    contamination = c_result.par[0] * np.ones((source.size, source.size)) + c_result.par[1] * np.sum(
        (flux_cube * flux_ratio)[np.array(list(set(np.arange(source.nstars)) ^ set(source.star_idx)))], axis=0)
    aperture = source.flux[num] - contamination

    r = PsfResult()
    r.linparam = list(c_result.par)
    r.nonlinparam = list(cfit)
    r.time = source.time[num]
    r.result = aperture
    return r
