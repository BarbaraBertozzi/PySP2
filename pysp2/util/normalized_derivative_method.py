import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

def central_difference(S, num_records=None):
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
    num_records: int or None
        Only process first num_records datapoints. Set to
        None to process all records.

    Returns
    -------
    dSdt : xarray Dataset
        Fourth-order numerical derivative S'(t).
    """
    
    if num_records is None:
        num_records = S.dims['event_index']
    
    dt = 200e-9  # Time step in seconds
    n_time = 100 # Number of time bins (assuming 100 bins from 0 to 100)

    # Process 'Data_ch0' and 'Data_ch4' only
    channels = ['Data_ch0', 'Data_ch4']
    dSdt = {}
    for ch in channels:
        dSdt_ch = np.full((num_records, n_time), np.nan, dtype=np.float64)
        for i_event in range(num_records):
            S_i = S.isel(event_index=i_event)
            y = S_i[ch].values
            dSdt_i = np.full_like(y, np.nan, dtype=np.float64)

            # --- Forward difference (i = 0, 1) ---
            dSdt_i[0] = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] - 3*y[4]) / (12*dt)
            dSdt_i[1] = (-25*y[1] + 48*y[2] - 36*y[3] + 16*y[4] - 3*y[5]) / (12*dt)

            # --- Central difference (i = 2 ... n-2) ---
            for i in range(2, n_time-2):
                dSdt_i[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12*dt)

            # --- Backward difference (i = n-2, n-1) ---
            dSdt_i[n_time-2] = (25*y[n_time-2] - 48*y[n_time-3] + 36*y[n_time-4] - 16*y[n_time-5] + 3*y[n_time-6]) / (12*dt)
            dSdt_i[n_time-1] = (25*y[n_time-1] - 48*y[n_time-2] + 36*y[n_time-3] - 16*y[n_time-4] + 3*y[n_time-5]) / (12*dt)
            
            dSdt_ch[i_event, :] = dSdt_i
        # Wrap as DataArray with event_index and time dims
        dSdt_ch0_ch4 = xr.DataArray(
            dSdt_ch,
            dims=('event_index', 'time'),
            coords={'event_index': S['event_index'], 'time': S['time']},
            name=f'd{ch}_dt'
        )
        dSdt[ch] = dSdt_ch0_ch4

    return xr.Dataset(dSdt)

