#!/usr/bin/env python

"""
This module defines a class called PsfFit for performing point spread function (PSF) fitting on astronomical images.
It follows the methodology outlined in Horne (1986) for optimal spectrum extraction:
https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H/abstract

This version is designed for IFU cubes, with the PSF fit smoothly across the wavelength dimension.

Version 2.0 includes PSF fitting.
"""

#############
#  Imports  #
#############
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

##############
# Authorship #
##############
__author__ = "Neelesh Amrutha"
__date__ = "23 January 2026"

__license__ = "GPL-3.0"
__version__ = "2.0"
__maintainer__ = "Neelesh Amrutha"
__email__ = "neelesh.amrutha<AT>anu.edu.au"

###############
#  Constants  #
###############
n_spec_bin = 8  # Number of spectral bins for PSF fitting
norm_factor = 1e13  # For better fitting


############################################################################################

def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    """Compute elliptical 2D Gaussian on grid xy.

    xy: tuple of (X, Y) meshgrid arrays (same shape)
    amp: amplitude
    x0, y0: center
    sigma_x, sigma_y: standard deviations along major/minor before rotation
    theta: rotation angle in radians (counterclockwise)
    offset: constant background
    """

    X, Y = xy
    x = X - x0
    y = Y - y0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # rotate coordinates
    xr = cos_t * x + sin_t * y
    yr = -sin_t * x + cos_t * y
    g = amp * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2)) + offset
    return g


def moffat_2d(xx, yy, x0, y0, alpha_x, alpha_y, beta, theta):
    """
    Elliptical, rotated Moffat profile on grid (xx, yy).
    Returns the Moffat evaluated at each pixel (not normalised to sum=1).
    Functional form:
       M(x,y) = [1 + R^2]^{-beta}
    where R^2 = (xr/alpha_x)^2 + (yr/alpha_y)^2 after rotation.
    """
    x = xx - x0
    y = yy - y0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # rotate coordinates (same convention used previously)
    xr = cos_t * x + sin_t * y
    yr = -sin_t * x + cos_t * y

    R2 = (xr / alpha_x)**2 + (yr / alpha_y)**2
    return (1.0 + R2) ** (-beta)


def make_model_image(params, x, y, data, model_type='gaussian'):
    """
    Create model image from params and compute residuals.
    """
    if model_type == 'gaussian':
        amp, x0, y0, sx, sy, theta, offset = params
        model = gaussian_2d((x, y), amp, x0, y0, sx, sy, theta, offset)

    elif model_type == 'moffat':
        amp, x0, y0, ax, ay, beta, theta, offset = params
        model = amp * moffat_2d(x, y, x0, y0, ax, ay, beta, theta) + offset

    else:
        raise ValueError("Unknown model_type")

    resid = data - model
    return resid, model


def _residuals(params, x, y, data, var, mask_valid, model_type='moffat'):
    """
    Residuals for least_squares. Uses sqrt(var) as sigma.

    Gaussian params:
      [amp, x0, y0, sigma_x, sigma_y, theta, offset]

    Moffat params:
      [amp, x0, y0, alpha, q, beta, theta, offset]
      Different parametrisation for fitting stability.
    """
    if model_type == 'moffat':
        # alpha and q to alpha_x, alpha_y
        amp, x0, y0, alpha, q, beta, theta, offset = params
        ax = alpha / np.sqrt(q)
        ay = alpha * np.sqrt(q)
        params = (amp, x0, y0, ax, ay, beta, theta, offset)

    resid, model = make_model_image(params, x, y, data, model_type)
    sigma = np.sqrt(var)

    # Return only properly weighted valid pixels
    r = resid[mask_valid] / np.maximum(sigma[mask_valid], 1e-12)  # whitened residuals
    return r.ravel()


def _make_spline(w, y, y_err, s, log=False, min_points=4):
    """
    Build an inverse-variance weighted smoothing spline.
    If log=True, spline is built in log(y) space (for widths).
    """
    w = np.asarray(w)
    y = np.asarray(y)
    y_err = np.asarray(y_err)
    m = (np.isfinite(w) & np.isfinite(y) & np.isfinite(y_err) & (y_err > 0))  # valid points

    if log:
        m &= (y > 0)

    if m.sum() < min_points:
        return None

    yy = np.log(y[m]) if log else y[m]

    # propagate errors for log-space
    if log:
        yy_err = y_err[m] / y[m]
    else:
        yy_err = y_err[m]

    wgt = 1.0 / yy_err**2
    spl = UnivariateSpline(w[m], yy, w=wgt, s=s)

    if not log:
        return spl

    # wrap spline so it returns exp(log α)
    def spl_exp(x):
        return np.exp(spl(x))

    return spl_exp


def fit_gaussian_2d(image, variance, x0_init, y0_init, box=10, p0=None, bounds=None):
    """
    Fit a 2D elliptical Gaussian + offset to image using variance as weights.

    Returns dict with best-fit params, uncertainties and fit metadata.
    """

    # Cutout
    ny, nx = image.shape
    xi = int(round(x0_init))
    yi = int(round(y0_init))

    x0 = max(0, xi - box)
    x1 = min(nx, xi + box + 1)
    y0 = max(0, yi - box)
    y1 = min(ny, yi + box + 1)

    cut_image = image[y0:y1, x0:x1].astype(float)
    cut_var = variance[y0:y1, x0:x1].astype(float)

    yy, xx = np.mgrid[0:cut_image.shape[0], 0:cut_image.shape[1]]
    X = xx + x0
    Y = yy + y0

    mask_valid = (np.isfinite(cut_image) & np.isfinite(cut_var) & (cut_var > 0))

    if mask_valid.sum() < 10:
        return {'success': False,
                'message': 'Too few valid pixels for fit',
                'params': {}, 'errors': {},
                'covariance': None, 'chi2': np.nan,
                'dof': 0, 'cutout_slice': (slice(y0, y1), slice(x0, x1))}

    # Initial parameters
    if p0 is None:
        amp0 = np.nanmax(cut_image[mask_valid]) - np.nanmedian(cut_image[mask_valid])
        amp0 = amp0 if amp0 > 0 else np.nanmax(cut_image[mask_valid])
        offset0 = np.nanmedian(cut_image[mask_valid])
        p0 = [amp0, x0_init, y0_init, 2.0, 2.0, 0.0, offset0]

    # Bounds
    if bounds is None:
        sigma_min = 1
        sigma_max = box * 0.8

        lower = [0.0, xi - 2, yi - 2, sigma_min, sigma_min, -np.pi, -1e5]
        upper = [1e5, xi + 2, yi + 2, sigma_max, sigma_max,  np.pi,  1e5]
        bounds = (lower, upper)

    # Least-squares fit
    res = least_squares(_residuals, x0=p0,
                        args=(X, Y, cut_image, cut_var, mask_valid, 'gaussian'),
                        bounds=bounds, method='trf')

    popt = res.x
    J = res.jac   # whitened Jacobian

    # Fit statistics
    n_pix = mask_valid.sum()
    dof = max(n_pix - len(popt), 1)

    chi2 = np.sum(res.fun ** 2)
    chi2_red = chi2 / dof

    # Robust covariance via SVD
    col_norm = np.linalg.norm(J, axis=0)
    good = col_norm > (1e-10 * col_norm.max())

    cov = np.full((len(popt), len(popt)), np.inf)

    if good.sum() > 0:
        Jr = J[:, good]

        U, s, VT = np.linalg.svd(Jr, full_matrices=False)
        thresh = np.finfo(float).eps * max(Jr.shape) * s[0]
        s_inv = np.array([1.0 / si if si > thresh else 0.0 for si in s])

        J_pinv = VT.T @ np.diag(s_inv) @ U.T
        cov_r = J_pinv @ J_pinv.T * chi2_red

        cov[np.ix_(good, good)] = cov_r

    p_err = np.sqrt(np.diag(cov))

    # Output
    keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
    params = dict(zip(keys, popt))
    errors = {k + '_err': e for k, e in zip(keys, p_err)}

    return {'params': params, 'errors': errors, 'covariance': cov,
            'success': bool(res.success), 'message': res.message,
            'chi2': chi2, 'chi2_red': chi2_red, 'dof': dof,
            'cutout_slice': (slice(y0, y1), slice(x0, x1)), }


def fit_moffat_2d(image, variance, x0_init, y0_init, box=10, p0=None, bounds=None):
    """
    Fit a 2D elliptical Moffat + offset to image using variance as weights.

    Returns dict with best-fit params, uncertainties and fit metadata.
    """

    # Cutout
    ny, nx = image.shape
    xi = int(round(x0_init))
    yi = int(round(y0_init))

    x0 = max(0, xi - box)
    x1 = min(nx, xi + box + 1)
    y0 = max(0, yi - box)
    y1 = min(ny, yi + box + 1)

    cut_image = image[y0:y1, x0:x1].astype(float)
    cut_var = variance[y0:y1, x0:x1].astype(float)

    yy, xx = np.mgrid[0:cut_image.shape[0], 0:cut_image.shape[1]]
    X = xx + x0
    Y = yy + y0

    # valid pixels
    mask_valid = (np.isfinite(cut_image) & np.isfinite(cut_var) & (cut_var > 0))

    if mask_valid.sum() < 10:
        return {'success': False, 'message': 'Too few valid pixels for fit',
                'params': {}, 'errors': {}, 'covariance': None,
                'chi2': np.nan, 'dof': 0, 'cutout_slice': (slice(y0, y1), slice(x0, x1))}

    # Initial parameters
    if p0 is None:
        amp0 = np.nanmax(cut_image[mask_valid]) - np.nanmedian(cut_image[mask_valid])
        amp0 = amp0 if amp0 > 0 else np.nanmax(cut_image[mask_valid])
        offset0 = np.nanmedian(cut_image[mask_valid])
        p0 = [amp0, x0_init, y0_init, 2.5, 1, 3, 0.0, offset0]

    if bounds is None:
        alpha_min, alpha_max = 1.0, 5.0  # Mean width scale
        q_min, q_max = 0.8, 1.2  # ellipticity parameter
        lower = [0.0, xi - 2, yi - 2, alpha_min, q_min, 1.1, -np.pi, -1e4]
        upper = [1e5, xi + 2, yi + 2, alpha_max, q_max, 5.0,  np.pi,  1e5]
        bounds = (lower, upper)

    # Least-squares fit
    res = least_squares(_residuals, x0=p0,
                        args=(X, Y, cut_image, cut_var, mask_valid, 'moffat'),
                        bounds=bounds, method='trf')

    popt = res.x
    J = res.jac   # whitened Jacobian

    # Degrees of freedom
    n_pix = mask_valid.sum()
    dof = max(n_pix - len(popt), 1)

    chi2 = np.sum(res.fun ** 2)
    chi2_red = chi2 / dof

    # Robust covariance via SVD
    col_norm = np.linalg.norm(J, axis=0)
    good = col_norm > (1e-10 * col_norm.max())
    cov = np.full((len(popt), len(popt)), np.inf)

    # Compute covariance for good parameters
    if good.sum() > 0:
        Jr = J[:, good]
        U, s, VT = np.linalg.svd(Jr, full_matrices=False)
        thresh = np.finfo(float).eps * max(Jr.shape) * s[0]
        s_inv = np.array([1.0 / si if si > thresh else 0.0 for si in s])
        J_pinv = VT.T @ np.diag(s_inv) @ U.T
        cov_r = J_pinv @ J_pinv.T * chi2_red
        cov[np.ix_(good, good)] = cov_r

    p_err = np.sqrt(np.diag(cov))

    # Output
    keys = ['amp', 'x0', 'y0', 'alpha', 'q', 'beta', 'theta', 'offset']

    # Convert to alpha_x, alpha_y
    amp, x0, y0, alpha, q, beta, theta, offset = popt
    sqrt_q = np.sqrt(q)
    alpha_err = p_err[keys.index('alpha')]
    q_err = p_err[keys.index('q')]
    alpha_x = alpha / sqrt_q
    alpha_y = alpha * sqrt_q
    alpha_x_err = np.sqrt((alpha_err / sqrt_q) ** 2 + (alpha * q_err / (2.0 * q ** (3 / 2))) ** 2)
    alpha_y_err = np.sqrt((alpha_err * sqrt_q) ** 2 + (alpha * q_err / (2.0 * sqrt_q)) ** 2)

    params = {'amp': amp, 'x0': x0, 'y0': y0,
              'alpha_x': alpha_x, 'alpha_y': alpha_y,
              'beta': beta, 'theta': theta, 'offset': offset}

    errors = {'amp_err': p_err[0], 'x0_err': p_err[1], 'y0_err': p_err[2],
              'alpha_x_err': alpha_x_err, 'alpha_y_err': alpha_y_err,
              'beta_err': p_err[5], 'theta_err': p_err[6], 'offset_err': p_err[7]}

    return {'params': params, 'errors': errors, 'covariance': cov,
            'success': bool(res.success), 'message': res.message,
            'chi2': chi2, 'chi2_red': chi2_red,
            'dof': dof, 'cutout_slice': (slice(y0, y1), slice(x0, x1))}


def evaluate_smooth_moffat(psf_spline_model, wavelength_axis):
    """
    Evaluate spectrally smooth Moffat PSF model at given wavelength domain.
    """
    x0 = psf_spline_model['x0'](wavelength_axis)
    y0 = psf_spline_model['y0'](wavelength_axis)
    ax = psf_spline_model['ax'](wavelength_axis)
    ay = psf_spline_model['ay'](wavelength_axis)
    be = psf_spline_model['beta'](wavelength_axis)
    th = psf_spline_model['th'](wavelength_axis)
    return x0, y0, ax, ay, be, th


def moffat_normalisation(alpha_x, alpha_y, beta):
    """
    Analytic integral of the elliptical Moffat over infinite plane:
      Integral M * dA = pi * alpha_x * alpha_y / (beta - 1)
    2D Moffat normalisation for elliptical axes.
    Returns the factor so that P = M / integral => sum(P) ~ 1 (continuum normalisation).
    """
    if beta <= 1.0:
        # pathological; return large value to avoid division by zero
        return np.inf
    return np.pi * alpha_x * alpha_y / (beta - 1.0)


def build_spectrally_smooth_psf_model(wave_centers, psf_fit_results):
    """
    Takes your PSF fits per spectral bin and builds smooth spline models
    for x0(λ), y0(λ), σx(λ), σy(λ), θ(λ)
    """
    x0 = np.array([r['params']['x0'] for r in psf_fit_results])
    y0 = np.array([r['params']['y0'] for r in psf_fit_results])
    sx = np.array([r['params']['sigma_x'] for r in psf_fit_results])
    sy = np.array([r['params']['sigma_y'] for r in psf_fit_results])
    th = np.array([r['params']['theta'] for r in psf_fit_results])

    # smoothing splines (s tuned for mild smoothing)
    spl_x0 = UnivariateSpline(wave_centers, x0, s=0.5)
    spl_y0 = UnivariateSpline(wave_centers, y0, s=0.5)
    spl_sx = UnivariateSpline(wave_centers, sx, s=10)
    spl_sy = UnivariateSpline(wave_centers, sy, s=10)
    spl_th = UnivariateSpline(wave_centers, th, s=0.5)

    return {'x0': spl_x0, 'y0': spl_y0, 'sx': spl_sx, 'sy': spl_sy, 'th': spl_th}


def build_spectrally_smooth_moffat_model(wave_centers, fit_results, beta_default=4.5,
                                         s_x0=8, s_y0=8, s_alpha=10, s_beta=8, s_theta=8):
    """
    Build UnivariateSplines for x0(λ), y0(λ), alpha_x(λ), alpha_y(λ), beta(λ), theta(λ).
    - If fit_results contains sigma_x/sigma_y (Gaussian fit), convert using beta_default.
    - Returns dictionary of spline objects keyed by 'x0','y0','ax','ay','beta','th'.
    """
    n = len(fit_results)
    x0 = np.zeros(n)
    y0 = np.zeros(n)
    ax = np.zeros(n)
    ay = np.zeros(n)
    be = np.zeros(n)
    th = np.zeros(n)

    for k, r in enumerate(fit_results):
        x0[k] = r['params'].get('x0', np.nan)
        y0[k] = r['params'].get('y0', np.nan)
        th[k] = r['params'].get('theta', 0.0)
        ax[k] = r['params'].get('alpha_x', np.nan)
        ay[k] = r['params'].get('alpha_y', np.nan)
        be[k] = r['params'].get('beta', beta_default)

    x0e = np.array([r['errors'].get('x0_err', np.inf) for r in fit_results])
    y0e = np.array([r['errors'].get('y0_err', np.inf) for r in fit_results])
    axe = np.array([r['errors'].get('alpha_x_err', np.inf) for r in fit_results])
    aye = np.array([r['errors'].get('alpha_y_err', np.inf) for r in fit_results])
    bee = np.array([r['errors'].get('beta_err', np.inf) for r in fit_results])
    the = np.array([r['errors'].get('theta_err', np.inf) for r in fit_results])

    # Build splines for position on 2D image
    spl_x0 = _make_spline(wave_centers, x0, x0e, s_x0)
    spl_y0 = _make_spline(wave_centers, y0, y0e, s_y0)

    # Sometimes alpha splines would shoot up randomly and I couldn't figure out why, so here's a dirty fix:
    dlog = np.abs(np.diff(np.log(ax)))
    ax[1:][dlog > 0.5] = np.nan  # Cannot have 3x jump between adjacent bins
    dlog = np.abs(np.diff(np.log(ay)))
    ay[1:][dlog > 0.5] = np.nan
    spl_ax = _make_spline(wave_centers, ax, axe, s_alpha, log=True)
    spl_ay = _make_spline(wave_centers, ay, aye, s_alpha, log=True)

    # Currently fixing beta and theta to median values, because they shouldn't change much with wavelength
    # comment out next two lines to fit splines
    be = np.ones(len(wave_centers)) * np.nanmedian(be)
    th = np.ones(len(wave_centers)) * np.nanmedian(th)
    spl_be = _make_spline(wave_centers, be, bee, s_beta)
    spl_th = _make_spline(wave_centers, th, the, s_theta)

    return {'x0': spl_x0, 'y0': spl_y0, 'ax': spl_ax, 'ay': spl_ay, 'beta': spl_be, 'th': spl_th}


def evaluate_smooth_psf(psf_spline_model, wavelength_axis):
    """
    Returns smooth Gaussian PSF parameters at every wavelength slice.
    """
    x0 = psf_spline_model['x0'](wavelength_axis)
    y0 = psf_spline_model['y0'](wavelength_axis)
    sx = psf_spline_model['sx'](wavelength_axis)
    sy = psf_spline_model['sy'](wavelength_axis)
    th = psf_spline_model['th'](wavelength_axis)
    return x0, y0, sx, sy, th


class PsfFit:

    def __init__(self, flux_cube, error_cube, init_row, init_col, model_type):
        # cubes are in (n_wave, n_row, n_col).
        self.flux = flux_cube
        self.error = error_cube  # Error must be variance
        self.n_wave, self.ny, self.nx = self.flux.shape

        # Scale for easier fitting
        self.flux *= norm_factor
        self.error *= norm_factor ** 2

        # Initial guess for PSF centre
        self.init_row = init_row
        self.init_col = init_col

        # Model type: 'gaussian' or 'moffat'
        self.model_type = model_type.lower()

        # Split spectrum into bins for PSF fitting
        self.flux_bins = np.array_split(self.flux, n_spec_bin, axis=0)
        self.error_bins = np.array_split(self.error, n_spec_bin, axis=0)

        self.extracted_spectrum = np.zeros(self.n_wave)
        self.extracted_error = np.zeros(self.n_wave)

        wave_centers = np.linspace(0, self.n_wave - 1, n_spec_bin)
        self.fit_results = None
        self.keys = None

        if self.model_type == 'gaussian':
            self.keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
            self.fit_results = self.fit_psf()
    
            # build smooth PSF model    
            psf_spline_model = build_spectrally_smooth_psf_model(
                wave_centers, self.fit_results)
    
            self.x0_arr, self.y0_arr, self.sx_arr, self.sy_arr, self.th_arr = evaluate_smooth_psf(
                psf_spline_model, np.arange(self.n_wave))
    
            self.extract_spectrum()

            # Rescale back to original units
            self.extracted_spectrum /= norm_factor
            self.extracted_error /= norm_factor
            
        elif self.model_type == 'moffat':
            self.keys = ['amp', 'x0', 'y0', 'alpha_x', 'alpha_y', 'beta', 'theta', 'offset']
            self.fit_results = self.fit_psf()

            # build smooth Moffat PSF model
            psf_spline_model = build_spectrally_smooth_moffat_model(wave_centers, self.fit_results)
            
            self.x0_arr, self.y0_arr, self.ax_arr, self.ay_arr, self.be_arr, self.th_arr = evaluate_smooth_moffat(
                psf_spline_model, np.arange(self.n_wave))
            
            self.extract_spectrum_moffat_weighted()

            # Rescale back to original units
            self.extracted_spectrum /= norm_factor
            self.extracted_error /= norm_factor

        else:
            raise ValueError("Unknown model_type. Choose 'gaussian' or 'moffat'.")

    def fit_psf(self):
        """
        Fit a PSF profile to each spectral bin.
        """
        fit_results = []

        for flux_bin, error_bin in zip(self.flux_bins, self.error_bins):
            # Sum over wavelength axis to get 2D image for PSF fitting
            image_2d = np.nansum(flux_bin, axis=0)
            error_2d = np.nansum(error_bin, axis=0)

            if self.model_type == 'gaussian':
                # Fit Gaussian PSF model to the 2D image
                fit_result = fit_gaussian_2d(image=image_2d, variance=error_2d,
                                             x0_init=self.init_col, y0_init=self.init_row, box=10)
            elif self.model_type == 'moffat':
                # Fit Moffat PSF model to the 2D image
                fit_result = fit_moffat_2d(image=image_2d, variance=error_2d,
                                           x0_init=self.init_col, y0_init=self.init_row, box=5)
            else:
                raise ValueError("Unknown model_type. Choose 'gaussian' or 'moffat'.")

            fit_results.append(fit_result)

        return fit_results

    def extract_spectrum(self):
        """
        Optimal extraction (Horne 1986) using a spectrally smooth Gaussian PSF model.

        self.flux  -> (n_wave, ny, nx)
        self.error -> (n_wave, ny, nx)  [VARIANCE cube]

        Returns:
            extracted_spectrum : (n_wave,)
            extracted_error    : (n_wave,)   [1-sigma uncertainty]
        """

        # wavelength-by-wavelength optimal extraction
        for i in range(self.n_wave):

            x0 = self.x0_arr[i]
            y0 = self.y0_arr[i]
            sx = self.sx_arr[i]
            sy = self.sy_arr[i]
            theta = self.th_arr[i]

            # extraction window (~2.5 sigma)
            rx = int(np.ceil(2.5 * sx))
            ry = int(np.ceil(2.5 * sy))

            x_min = max(0, int(x0) - rx)
            x_max = min(self.nx, int(x0) + rx + 1)
            y_min = max(0, int(y0) - ry)
            y_max = min(self.ny, int(y0) + ry + 1)

            # extract data + variance
            D = self.flux[i, y_min:y_max, x_min:x_max]  # DATA
            V = self.error[i, y_min:y_max, x_min:x_max]  # VARIANCE

            if not np.any(np.isfinite(D)):
                self.extracted_spectrum[i] = np.nan
                self.extracted_error[i] = np.nan
                continue

            # coordinate grid
            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

            # Gaussian PSF model
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            xr = cos_t * (xx - x0) + sin_t * (yy - y0)
            yr = -sin_t * (xx - x0) + cos_t * (yy - y0)
            P = (1 / (2 * np.pi * sx * sy)) * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))  # PSF PROFILE

            # valid variance mask
            mask = np.isfinite(D) & np.isfinite(V) & (V > 0)

            if np.sum(mask) < 5:
                self.extracted_spectrum[i] = np.nan
                self.extracted_error[i] = np.nan
                continue

            Dm = D[mask]
            Vm = V[mask]
            Pm = P[mask]

            # Optimal (Horne 1986) estimator
            numerator = np.sum(Pm * Dm / Vm)
            denominator = np.sum(Pm ** 2 / Vm)

            if denominator <= 0:
                self.extracted_spectrum[i] = np.nan
                self.extracted_error[i] = np.nan
                continue

            self.extracted_spectrum[i] = numerator / denominator
            self.extracted_error[i] = 1.0 / np.sqrt(denominator)

    def extract_spectrum_moffat_weighted(self, beta_default=4.5, trunc_sigma=2.5):
        """
        Optimal extraction (Horne 1986) using a spectrally-smooth Moffat PSF.
        - self.flux  : (n_wave, ny, nx)
        - self.error : (n_wave, ny, nx)  [VARIANCE]
        Returns:
          extracted_flux, extracted_error (1-sigma)
        """

        # build smooth PSF model
        wave_centers = np.linspace(0, self.n_wave - 1, n_spec_bin)
        psf_model = build_spectrally_smooth_moffat_model(wave_centers, self.fit_results, beta_default=beta_default)
        x0_arr, y0_arr, ax_arr, ay_arr, be_arr, th_arr = evaluate_smooth_moffat(psf_model, np.arange(self.n_wave))

        for i in range(self.n_wave):
            x0 = x0_arr[i]
            y0 = y0_arr[i]
            alpha_x = ax_arr[i]
            alpha_y = ay_arr[i]
            beta = be_arr[i]
            theta = th_arr[i]

            # approximate 'sigma' scale to set truncation window:
            # use equivalent gaussian sigma ~ alpha * sqrt(1/(2^(1/beta) -1)) / 2.355? simpler: estimate sigma_eff
            # but keep truncation in units of alpha_x/alpha_y:
            rx = int(np.ceil(trunc_sigma * alpha_x))
            ry = int(np.ceil(trunc_sigma * alpha_y))

            x_min = max(0, int(x0) - rx)
            x_max = min(self.nx, int(x0) + rx + 1)
            y_min = max(0, int(y0) - ry)
            y_max = min(self.ny, int(y0) + ry + 1)

            D = self.flux[i, y_min:y_max, x_min:x_max]
            V = self.error[i, y_min:y_max, x_min:x_max]

            if not np.any(np.isfinite(D)):
                self.extracted_spectrum[i] = np.nan
                self.extracted_error[i] = np.nan
                continue

            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

            M = moffat_2d(xx, yy, x0, y0, alpha_x, alpha_y, beta, theta)

            # analytic norm (continuous integral) for elliptical moffat:
            integral = moffat_normalisation(alpha_x, alpha_y, beta)

            # PROFILE
            if np.isfinite(integral) and integral > 0:
                P = M / integral  # continuous normalization (sum(P) ~ 1)
            else:
                # fallback: normalize by discrete sum
                P_raw = M
                P_sum = np.nansum(P_raw)
                if P_sum <= 0:
                    self.extracted_spectrum[i] = np.nan
                    self.extracted_error[i] = np.nan
                    continue
                P = P_raw / P_sum
                print(f"Warning: Moffat PSF normalisation fallback to discrete sum, idx={i}")

            mask = np.isfinite(D) & np.isfinite(V) & (V > 0)
            if np.sum(mask) < 5:
                self.extracted_spectrum[i] = np.nan
                self.extracted_error[i] = np.nan
                continue

            Dm = D[mask]
            Vm = V[mask]
            Pm = P[mask]

            numerator = np.sum(Pm * Dm / Vm)
            denominator = np.sum(Pm ** 2 / Vm)

            if denominator <= 0:
                self.extracted_spectrum[i] = np.nan
                self.extracted_error[i] = np.nan
                continue

            self.extracted_spectrum[i] = numerator / denominator
            self.extracted_error[i] = 1.0 / np.sqrt(denominator)

    def make_model_evaluation_plot(self, save=None):
        """
        Make diagnostic plot of PSF model parameters vs wavelength bin centers.
        """

        fig, axes = plt.subplots(8, 3, figsize=(6, 16))

        # Generate and plot a PSF model for each spectral bin based on the fit
        for i in range(8):
            yy, xx = np.mgrid[0:self.ny, 0:self.nx]
            image_2d = np.nansum(self.flux_bins[i], axis=0)

            params = [self.fit_results[i]['params'][key] for key in self.keys]

            resid, model = make_model_image(params, xx, yy, image_2d, model_type=self.model_type)
            # normalise residual to error
            # resid_norm = resid / np.maximum(error_2d, 1e-12)
            resid_max = np.max(np.abs(resid)) * 0.8

            im = axes[i, 0].imshow(image_2d, origin='lower', cmap='viridis')
            plt.colorbar(im, ax=axes[i, 0], fraction=0.046*35/25, pad=0.04)
            c_lim = im.get_clim()
            im = axes[i, 1].imshow(model, origin='lower', cmap='viridis')
            im.set_clim(c_lim)
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046*35/25, pad=0.04)
            # Forcing centered colourmap for residuals
            im = axes[i, 2].imshow(resid, origin='lower', cmap='bwr', vmin=-resid_max, vmax=resid_max)
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046*35/25, pad=0.04)

        axes[0, 0].set_title('Data')
        axes[0, 1].set_title('PSF Model')
        axes[0, 2].set_title('Residual 80%')
        plt.tight_layout()
        if save:
            plt.savefig(save)

        plt.close()

    def get_single_fit_plot(self, bin_psf=3):
        """
        Generate data, model, residual images for a single PSF fit bin.
        """
        if self.fit_results:
            result = self.fit_results[bin_psf]

            params = [result['params'][key] for key in self.keys]
            yy, xx = np.mgrid[0:self.ny, 0:self.nx]
            image_2d = np.nansum(self.flux_bins[bin_psf], axis=0)

            resid, model = make_model_image(params, xx, yy, image_2d, model_type=self.model_type)
            image_2d /= norm_factor
            resid /= norm_factor
            model /= norm_factor

            return image_2d, resid, model
        else:
            return None
