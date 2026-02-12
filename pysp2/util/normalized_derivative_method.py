import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

def central_difference(S, min_signal=1e-13, num_records=None):
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
    num_records: int or None
        Only process first num_records datapoints. Set to
        None to process all records.

    Returns
    -------
    dSdt : xarray Dataset
        Fourth-order numerical derivative S'(t).
    norm_deriv: xarray Dataset
        Normalized derivative S'(t) / S(t).
    """
    
    if num_records is None:
        num_records = S.dims['event_index']
    print(f'Processing {num_records} records...')
    
    time = S['time'].values
    print(f'Time:' f'  shape: {time.shape}, dtype: {time.dtype}, min: {time.min()}, max: {time.max()}')
    if time.ndim == 2:
        time = time[0]  # Take the first event's time axis if 2D
    dt = np.diff(time).mean()
    if np.issubdtype(dt.dtype, np.timedelta64):
        dt = dt / np.timedelta64(1, 's')  # convert to seconds as float
    print(f'Calculated dt: {dt}')
    if dt == 0 or np.isnan(dt):
        print('Warning: dt is zero or NaN!')
    n_time = 100
    print(f'Number of time points: {n_time}')
    # Only process the first record and 'Data_ch0' for now
    S_i = S.isel(event_index=0)
    ch = 'Data_ch0'
    y = S_i[ch].values
    print(f'Input signal y: shape: {y.shape}, dtype: {y.dtype}, min: {y.min()}, max: {y.max()}')    
    dSdt = np.full_like(y, np.nan, dtype=np.float64)
    # --- Forward difference (i = 0, 1) ---
    dSdt[0] = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4]) / (12*dt)
    dSdt[1] = (-25*y[1] + 48*y[2] - 36*y[3] + 16*y[4] - 3*y[5]) / (12*dt)
    # --- Central difference (i = 2 ... n-3) ---
    for i in range(2, n_time-2):
        dSdt[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12*dt)
    # --- Backward difference (i = n-2, n-1) ---
    dSdt[n_time-2] = (25*y[n_time-2] - 48*y[n_time-3] + 36*y[n_time-4] - 16*y[n_time-5] + 3*y[n_time-6]) / (12*dt)
    dSdt[n_time-1] = (25*y[n_time-1] - 48*y[n_time-2] + 36*y[n_time-3] - 16*y[n_time-4] + 3*y[n_time-5]) / (12*dt)
    # Wrap as DataArray with same coords as input
    dSdt_da = xr.DataArray(dSdt, dims=S_i[ch].dims, coords=S_i[ch].coords, name=f'd{ch}_dt')
    return dSdt_da

