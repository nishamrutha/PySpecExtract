#!/usr/bin/env python

"""
This module defines a class called PsfFit for performing point spread function (PSF) fitting on astronomical images.


Version 2.0 includes psf fitting.
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
__date__ = "03 December 2025"

__license__ = "GPL-3.0"
__version__ = "2.0"
__maintainer__ = "Neelesh Amrutha"
__email__ = "neelesh.amrutha<AT>anu.edu.au"

###############
#  Constants  #
###############
n_spec_bin = 8  # Number of spectral bins for PSF fitting
norm_factor = 1e15  # For better fitting


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
    Returns the Moffat evaluated at each pixel (not normalized to sum=1).
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

    resid = model - data
    return resid, model


def _residuals(params, x, y, data, var, mask_valid, model_type='gaussian'):
    """
    Residuals for least_squares. Uses sqrt(var) as sigma.

    Gaussian params:
      [amp, x0, y0, sigma_x, sigma_y, theta, offset]

    Moffat params:
      [amp, x0, y0, alpha_x, alpha_y, beta, theta, offset]
    """
    resid, model = make_model_image(params, x, y, data, model_type)
    sigma = np.sqrt(var)

    # Return only properly weighted valid pixels
    r = resid[mask_valid] / np.maximum(sigma[mask_valid], 1e-12)
    return r.ravel()


def fit_gaussian_2d(image, variance, x0_init, y0_init, box=10, p0=None, bounds=None):
    """Fit a 2D elliptical Gaussian + offset to image using variance as weights.

    image, variance: 2D numpy arrays
    x0_init, y0_init: center in pixel coordinates (image pixel frame)
    box: half-width of fitting box (so size = 2*box+1)
    p0: optional initial parameters [amp, x0, y0, sigma_x, sigma_y, theta, offset]
    bounds: optional bounds tuple (lower, upper) for parameters

    Returns dict with best-fit params, uncertainties (approx from Jacobian), and fit metadata.
    """

    # cutout
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
    # convert to same pixel coordinates as original
    X = xx + x0
    Y = yy + y0

    # mask valid pixels where variance > 0 and finite
    mask_valid = np.isfinite(cut_image) & np.isfinite(cut_var) & (cut_var > 0)

    # initial params
    if p0 is None:
        amp0 = np.nanmax(cut_image) - np.nanmedian(cut_image)
        amp0 = amp0 if amp0 > 0 else np.nanmax(cut_image)
        offset0 = np.nanmedian(cut_image)
        p0 = [amp0, x0_init, y0_init, 2, 2, 0.0, offset0]

    if bounds is None:
        lower = [0.0, xi - 2, yi - 2, 0.3, 0.3, -np.pi, -1e4]
        upper = [1e4, xi + 2, yi + 2, box, box, np.pi, 1e4]
        bounds = (lower, upper)

    # flatten mask for selection in residual function
    mask_flat = mask_valid

    res = least_squares(_residuals, x0=p0, args=(X, Y, cut_image, cut_var, mask_flat), bounds=bounds)

    popt = res.x
    # estimate covariance matrix: cov ~= inv(J^T J) * residual_variance
    # note: least_squares returns jac scaled; compute residual variance per dof
    J = res.jac
    _, s, VT = np.linalg.svd(J, full_matrices=False)
    # pseudo-inverse
    threshold = np.finfo(float).eps * max(J.shape) * s[0]
    s_inv = np.array([1.0 / si if si > threshold else 0.0 for si in s])
    J_pinv = (VT.T * s_inv) @ (np.eye(len(s))) @ J.T
    dof = np.sum(mask_valid) - len(popt)
    dof = float(dof) if dof > 0 else 1.0
    resid = res.fun
    resid_var = np.sum(resid ** 2) / dof
    try:
        cov = J_pinv @ J_pinv.T * resid_var
        p_err = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        cov = None
        p_err = np.full_like(popt, np.nan)

    keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
    params = {k: float(v) for k, v in zip(keys, popt)}
    errors = {k + '_err': float(e) for k, e in zip(keys, p_err)}

    out = {
        'params': params,
        'errors': errors,
        'covariance': cov,
        'success': bool(res.success),
        'message': res.message,
        'chi2': float(np.sum(res.fun ** 2)),
        'dof': int(np.sum(mask_valid) - len(popt)),
        'cutout_slice': (slice(y0, y1), slice(x0, x1)),
    }
    return out


def fit_moffat_2d(image, variance, x0_init, y0_init, box=10, p0=None, bounds=None):
    """Fit a 2D elliptical Moffat + offset to image using variance as weights.

    image, variance: 2D numpy arrays
    x0_init, y0_init: center in pixel coordinates (image pixel frame)
    box: half-width of fitting box (so size = 2*box+1)
    p0: optional initial parameters [amp, x0, y0, alpha_x, alpha_y, beta, theta, offset]
    bounds: optional bounds tuple (lower, upper) for parameters

    Returns dict with best-fit params, uncertainties (approx from Jacobian), and fit metadata.
    """

    # cutout
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
    # convert to same pixel coordinates as original
    X = xx + x0
    Y = yy + y0

    # mask valid pixels where variance > 0 and finite
    mask_valid = np.isfinite(cut_image) & np.isfinite(cut_var) & (cut_var > 0)

    # initial params
    if p0 is None:
        amp0 = np.nanmax(cut_image) - np.nanmedian(cut_image)
        amp0 = amp0 if amp0 > 0 else np.nanmax(cut_image)
        offset0 = np.nanmedian(cut_image)
        p0 = [amp0, x0_init, y0_init, 2, 2, 4.5, 0.0, offset0]

    if bounds is None:
        lower = [0.0, xi - 2, yi - 2, 0.3, 0.3, 1.12, -np.pi, -1e2]
        upper = [1e4, xi + 2, yi + 2, box, box, 5, np.pi, 1e4]
        bounds = (lower, upper)

    # least squares fit
    res = least_squares(_residuals, x0=p0,
                        args=(X, Y, cut_image, cut_var, mask_valid, 'moffat'),
                        bounds=bounds)

    popt = res.x
    J = res.jac

    # correct SVD-based pseudoinverse
    U, s, VT = np.linalg.svd(J, full_matrices=False)
    threshold = np.finfo(float).eps * max(J.shape) * s[0]
    s_inv = np.array([1.0 / si if si > threshold else 0.0 for si in s])
    J_pinv = VT.T @ np.diag(s_inv) @ U.T

    dof = np.sum(mask_valid) - len(popt)
    dof = float(dof) if dof > 0 else 1.0

    resid = res.fun
    resid_var = np.sum(resid ** 2) / dof

    try:
        cov = J_pinv @ J_pinv.T * resid_var
        p_err = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        cov = None
        p_err = np.full_like(popt, np.nan)

    keys = ['amp', 'x0', 'y0', 'alpha_x', 'alpha_y', 'beta', 'theta', 'offset']
    params = {k: float(v) for k, v in zip(keys, popt)}
    errors = {k + '_err': float(e) for k, e in zip(keys, p_err)}

    out = {
        'params': params,
        'errors': errors,
        'covariance': cov,
        'success': bool(res.success),
        'message': res.message,
        'chi2': float(np.sum(res.fun ** 2)),
        'dof': int(np.sum(mask_valid) - len(popt)),
        'cutout_slice': (slice(y0, y1), slice(x0, x1)),
    }
    return out


def evaluate_smooth_moffat(psf_spline_model, wavelength_axis):
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
      Integral M dA = pi * alpha_x * alpha_y / (beta - 1)
    See: 2D Moffat normalisation for elliptical axes.
    We return the factor so that P = M / integral => sum(P) ~ 1 (continuum normalisation).
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
    spl_sx = UnivariateSpline(wave_centers, sx, s=0.3)
    spl_sy = UnivariateSpline(wave_centers, sy, s=0.3)
    spl_th = UnivariateSpline(wave_centers, th, s=0.3)

    return {
        'x0': spl_x0,
        'y0': spl_y0,
        'sx': spl_sx,
        'sy': spl_sy,
        'th': spl_th
    }


def build_spectrally_smooth_moffat_model(wave_centers, fit_results, beta_default=4.5,
                                         s_x0=0.5, s_y0=0.5, s_alpha=0.2, s_beta=0.2, s_theta=0.2):
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

    # make splines; choose smoothing 's' values loosely tuned (you can tweak)
    spl_x0 = UnivariateSpline(wave_centers, x0, s=s_x0)
    spl_y0 = UnivariateSpline(wave_centers, y0, s=s_y0)
    spl_ax = UnivariateSpline(wave_centers, ax, s=s_alpha)
    spl_ay = UnivariateSpline(wave_centers, ay, s=s_alpha)
    spl_be = UnivariateSpline(wave_centers, be, s=s_beta)
    spl_th = UnivariateSpline(wave_centers, th, s=s_theta)

    return {
        'x0': spl_x0,
        'y0': spl_y0,
        'ax': spl_ax,
        'ay': spl_ay,
        'beta': spl_be,
        'th': spl_th
    }


def evaluate_smooth_psf(psf_spline_model, wavelength_axis):
    """
    Returns smooth PSF parameters at every wavelength slice.
    """

    x0 = psf_spline_model['x0'](wavelength_axis)
    y0 = psf_spline_model['y0'](wavelength_axis)
    sx = psf_spline_model['sx'](wavelength_axis)
    sy = psf_spline_model['sy'](wavelength_axis)
    th = psf_spline_model['th'](wavelength_axis)

    return x0, y0, sx, sy, th


class PsfFit:

    def __init__(self, flux_cube, error_cube, init_row, init_col, model_type):
        # cubes are in (n_wave, n_row, n_col) format.
        self.flux = flux_cube
        self.error = error_cube  # Error must be variance
        self.n_wave, self.ny, self.nx = self.flux.shape

        # Scale for easier fitting
        self.flux *= norm_factor
        self.error *= norm_factor**2
        print(np.median(self.flux), np.median(self.error))

        # Initial guess for PSF center
        self.init_row = init_row
        self.init_col = init_col

        # Model type: 'gaussian' or 'moffat'
        self.model_type = model_type.lower()

        # Split spectrum into bins for PSF fitting
        # self.wave_bins = np.array_split(self.wave, n_spec_bin)
        self.flux_bins = np.array_split(self.flux, n_spec_bin, axis=0)
        self.error_bins = np.array_split(self.error, n_spec_bin, axis=0)

        self.extracted_spectrum = np.zeros(self.n_wave)
        self.extracted_error = np.zeros(self.n_wave)

        wave_centers = np.linspace(0, self.n_wave - 1, n_spec_bin)

        if self.model_type == 'gaussian':
            self.fit_results = self.fit_psf()
    
            # build smooth PSF model    
            psf_spline_model = build_spectrally_smooth_psf_model(
                wave_centers,
                self.fit_results
            )
    
            self.x0_arr, self.y0_arr, self.sx_arr, self.sy_arr, self.th_arr = evaluate_smooth_psf(
                psf_spline_model,
                np.arange(self.n_wave)
            )
    
            self.extract_spectrum()

            # Rescale back to original units
            self.extracted_spectrum /= norm_factor
            self.extracted_error /= norm_factor
            
        elif self.model_type == 'moffat':
            self.fit_results = self.fit_moffat_psf()

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
        fit_results = []

        for flux_bin, error_bin in zip(self.flux_bins, self.error_bins):
            # Sum over wavelength axis to get 2D image for PSF fitting
            image_2d = np.nansum(flux_bin, axis=0)
            error_2d = np.nansum(error_bin, axis=0)

            # Fit PSF model to the 2D image
            fit_result = fit_gaussian_2d(
                image=image_2d,
                variance=error_2d,
                x0_init=self.init_col,
                y0_init=self.init_row,
                box=10
            )

            fit_results.append(fit_result)

        return fit_results
    
    def fit_moffat_psf(self):
        fit_results = []

        for flux_bin, error_bin in zip(self.flux_bins, self.error_bins):
            # Sum over wavelength axis to get 2D image for PSF fitting
            image_2d = np.nansum(flux_bin, axis=0)
            error_2d = np.nansum(error_bin, axis=0)  # Variance here

            # Fit Moffat PSF model to the 2D image
            fit_result = fit_moffat_2d(
                image=image_2d,
                variance=error_2d,
                x0_init=self.init_col,
                y0_init=self.init_row,
                box=10)
            fit_results.append(fit_result)
        return fit_results

    def extract_spectrum(self):
        """
        True PSF-weighted optimal extraction using a spectrally smooth PSF model.

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
            D = self.flux[i, y_min:y_max, x_min:x_max]
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

            P = (1 / (2 * np.pi * sx * sy)) * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

            # P = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
            # normalize PSF
            # P_sum = np.nansum(P)
            # if P_sum <= 0:
            #     self.extracted_spectrum[i] = np.nan
            #     self.extracted_error[i] = np.nan
            #     continue
            # P = P / P_sum  # discrete normalization

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
        Optimal extraction using a spectrally-smooth Moffat PSF.
        - self.flux  : (n_wave, ny, nx)
        - self.error : (n_wave, ny, nx)  [VARIANCE]
        Returns:
          extracted_flux, extracted_error (1-sigma)
        """

        # build smooth model
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

        if self.model_type == 'gaussian':
            keys = ['amp', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
        elif self.model_type == 'moffat':
            keys = ['amp', 'x0', 'y0', 'alpha_x', 'alpha_y', 'beta', 'theta', 'offset']
        else:
            keys = None

        # Generate and plot a psf model for each spectral bin based on the fit
        for i in range(8):
            yy, xx = np.mgrid[0:self.ny, 0:self.nx]
            image_2d = np.nansum(self.flux_bins[i], axis=0)
            error_2d = np.sqrt(np.nansum(self.error_bins[i], axis=0))

            params = [self.fit_results[i]['params'][key] for key in keys]

            resid, model = make_model_image(params, xx, yy, image_2d, model_type=self.model_type)
            # normalise residual to error
            resid_norm = resid / np.maximum(error_2d, 1e-12)
            resid_max = np.max(np.abs(resid_norm)) * 0.8

            im = axes[i, 0].imshow(image_2d, origin='lower', cmap='viridis')
            plt.colorbar(im, ax=axes[i, 0], fraction=0.046*35/25, pad=0.04)
            c_lim = im.get_clim()
            im = axes[i, 1].imshow(model, origin='lower', cmap='viridis')
            im.set_clim(c_lim)
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046*35/25, pad=0.04)
            im = axes[i, 2].imshow(resid_norm, origin='lower', cmap='bwr', vmin=-resid_max, vmax=resid_max)
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046*35/25, pad=0.04)

        axes[0, 0].set_title('Data')
        axes[0, 1].set_title('PSF Model')
        axes[0, 2].set_title('Residual 80%')
        plt.tight_layout()
        if save:
            plt.savefig(save)

        plt.close()
