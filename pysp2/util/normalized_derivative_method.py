import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

def central_difference_scheme(S, min_signal=1e-12):
    """
    Compute normalized derivative S'(t) / S(t) using the
    central difference scheme (Moteki & Kondo, Eq. A.2).
    - Fourth-order central difference
    Edge cases:
    - Fourth-order forward difference at the beginning
    - Fourth-order backward difference at the end

    Parameters
    ----------
    S: xarray Dataset
        The scattering signal dataset.
    min_signal: float
        Minimum S(t) signal value to avoid division by zero.

    Returns
    -------
    dSdt : ndarray
        Fourth-order numerical derivative S'(t).
    norm_deriv: xarray Dataset
        Normalized derivative S'(t) / S(t).
    """
    spectra = ds.isel(

    return dSdt, norm_deriv