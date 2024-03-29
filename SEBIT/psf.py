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
        self.flux_cube = None


class Mesh(object):
    """
    Building 3d meshgrids of star information

    Attributes
    ----------
    # TODO: comment
    """

    def __init__(self, source):
        if source.gaia_cut is None:
            gaia = source.gaia
            size = source.size
        else:
            gaia = source.gaia_cut
            size = 11
        x, y, flux_ratio = np.meshgrid(np.arange(size), np.arange(size),
                                       gaia['tess_flux_ratio'][0:source.nstars])
        x_shift = np.meshgrid(np.zeros(size), np.zeros(size),
                              np.array(gaia['Sector_{}_x'.format(source.sector)])[0:source.nstars])[2]
        y_shift = np.meshgrid(np.zeros(size), np.zeros(size),
                              np.array(gaia['Sector_{}_y'.format(source.sector)])[0:source.nstars])[2]
        self.x = x.transpose(2, 0, 1)
        self.y = y.transpose(2, 0, 1)
        self.flux_ratio = flux_ratio.transpose(2, 0, 1)
        self.x_shift = x_shift.transpose(2, 0, 1)
        self.y_shift = y_shift.transpose(2, 0, 1)
        self.A = np.ones((len(source.z), 2 + len(source.star_index)))
        self.size = size


class Star(object):
    def __init__(self, index, source, psf_result):
        corr_flux = psf_result[:, 2]
        gaia = source.gaia
        if source.gaia_cut is None:
            self.source = gaia[index]
        else:
            idx = np.where(source.gaia_cut['designation'] == gaia['designation'][index])
            self.source = source.gaia_cut[idx]
        self.flux = corr_flux
        # x = int(self.source[f'Sector_{source.sector}_x'])
        # y = int(self.source[f'Sector_{source.sector}_y'])
        # image = np.dstack(psf_result[:, -1])[x - 1:x + 1, y - 1:y + 1, :]
        # flux = np.sum(np.sum(image, axis=0), axis=0)
        # self.flux = flux / psf_result[:, 1]
        self.cnn = 0.5
        self.period = 1


def chisq_model(par, model, flux, source, mesh):
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
    return np.sum((model(par, flux, source, mesh).y - flux) ** 2)

    # TODO: set a moffat boundary?


def moffat_model(c, flux, source, mesh):
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
    x = mesh.x_shift + c[0]
    y = mesh.y_shift + c[1]
    x_shift = x * np.cos(c[2]) - y * np.sin(c[2])
    y_shift = x * np.sin(c[2]) + y * np.cos(c[2])
    x = mesh.x - x_shift
    y = mesh.y - y_shift
    flux_cube = (1 + (c[3] * x ** 2 + 2 * c[4] * x * y + c[5] * y ** 2)) ** (- c[6])
    A = mesh.A
    A[:, 0] = 1  # F_bg
    A[:, 1] = np.sum(
        (flux_cube * mesh.flux_ratio)[np.array(list(set(np.arange(source.nstars)) ^ set(source.star_index)))],
        axis=0).reshape(mesh.size ** 2)  # F_norm
    for j, index in enumerate(source.star_index):
        A[:, j + 2] = flux_cube[index].reshape(mesh.size ** 2)  # F_ebs

    result = Linmodel()
    if np.isnan(np.sum(A)):
        result.y = np.inf
        return result
        # making result.par = None
    fluxfit = np.zeros(len(source.z))
    Ap = A
    bp = flux
    fit = np.linalg.lstsq(Ap, bp, rcond=None)[0]
    fluxfit += np.dot(A, fit)
    result.y = fluxfit
    result.par = fit
    result.flux_cube = flux_cube
    return result


def psf(source, num=0, c=None, aperture=False, mesh=None):
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
    aperture: Boolean
    mesh:
    """
    if num == -1:
        flux = np.mean(source.flux, axis=0)
    elif source.gaia_cut is None:
        flux = source.flux[num]
    elif num == -2:
        flux = np.mean(source.flux_cut, axis=0)
    else:
        flux = source.flux_cut[num]
    flux_flat = flux.reshape(mesh.size ** 2)
    if c is None:
        cfit = optimize.minimize(chisq_model, source.cguess, (moffat_model, flux_flat, source, mesh),
                                 bounds=source.var_to_bounds, method="Powell").x
        # options={'disp': True}

        # TODO: method?
    else:
        cfit = c
    c_result = moffat_model(cfit, flux_flat, source, mesh)

    r = list(c_result.par)
    r.extend(list(cfit))
    r.append(source.time[num])
    if aperture is True:
        flux_cube = c_result.flux_cube
        contamination = c_result.par[0] * np.ones((mesh.size, mesh.size)) + c_result.par[1] * np.sum(
            (flux_cube * mesh.flux_ratio)[np.array(list(set(np.arange(source.nstars)) ^ set(source.star_index)))], axis=0)
        aperture = flux - contamination
        r.append(aperture)
    return r
