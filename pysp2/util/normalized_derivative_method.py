import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

def central_difference(S, min_signal=1e-12, num_records=None):
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
    
    time = S['time'].values
    if time.ndim == 2:
        time = time[0]  # Take the first event's time axis if 2D
    dt = 1
    n_time = 1
    for i_records in range(num_records):
        S_i = S.isel(event_index=i_records)
        S_i = S_i[['Data_ch0', 'Data_ch4']]
        #S_i['Data_ch0'] = S_i['Data_ch0'].clip(min=min_signal)
        #S_i['Data_ch4'] = S_i['Data_ch4'].clip(min=min_signal)

        dSdt = xr.full_like(S_i, fill_value=np.nan)

        # --- Forward difference (i = 0, 1) ---
        dSdt[0] = (
            -25*S_i[0] + 48*S_i[1] - 36*S_i[2] + 16*S_i[3] - 3*S_i[4]
        ) / (12*dt)

        dSdt[1] = (
            -25*S_i[1] + 48*S_i[2] - 36*S_i[3] + 16*S_i[4] - 3*S_i[5]
        ) / (12*dt)

        # --- Central difference (i = 2 ... n-3) ---
        for i in range(2, n_time-2):
            dSdt[i] = (
                -S_i[i+2] + 8*S_i[i+1]
                - 8*S_i[i-1] + S_i[i-2]
            ) / (12*dt)

        # --- Backward difference (i = n-2, n-1) ---
        dSdt[n_time-2] = (
            25*S_i[n_time-2] - 48*S_i[n_time-3]
            + 36*S_i[n_time-4] - 16*S_i[n_time-5]
            + 3*S_i[n_time-6]
        ) / (12*dt)

        dSdt[n_time-1] = (
            25*S_i[n_time-1] - 48*S_i[n_time-2]
            + 36*S_i[n_time-3] - 16*S_i[n_time-4]
            + 3*S_i[n_time-5]
        ) / (12*dt)

        # Compute the normalized derivative
        norm_deriv = dSdt / S_i

        dSdt[i_records, :] = dSdt
        norm_deriv[i_records, :] = norm_deriv
        print(f'Processed record {i_records+1}/{num_records}')
        print(f'Min S(t) = {S_i.min().values}, Max S(t) = {S_i.max().values}')
        print(f'Min dS/dt = {dSdt.min().values}, Max dS/dt = {dSdt.max().values}')
        print(f'Min normalized derivative = {norm_deriv.min().values}, Max normalized derivative = {norm_deriv.max().values}')

    return dSdt, norm_deriv

