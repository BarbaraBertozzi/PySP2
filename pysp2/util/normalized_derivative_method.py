import numpy as np
import xarray as xr

def central_difference(S, num_records=None):

    """ 
    Compute fourth order derivative S'(t) using the 
    central difference scheme (Moteki & Kondo, Eq. A.1)
    for scattering channels (ch0 and ch4).
    Interior points:
        - Fourth-order central difference 
    Edge cases: 
        - First two points: fourth-order forward difference  
        - Last two points: fourth-order backward difference 

    Parameters 
    ---------- 
    S: xarray Dataset 
        The scattering signal dataset. 
    num_records: int or None 
        Only process first num_records datapoints. 
        Set to None to process all records. 

    Returns 
    ------- 
    dSdt : xarray Dataset 
        Fourth-order numerical derivative S'(t). 
    """

    if num_records is None:
        num_records = S.sizes['event_index']

    dt = 200e-9
    channels = ['Data_ch0', 'Data_ch4']
    dSdt = {}

    for ch in channels:
        y = S[ch].isel(event_index=slice(0, num_records)).values
        d = np.full_like(y, np.nan, dtype=np.float64)

        # Interior points (vectorized)
        d[:, 2:-2] = (
            -y[:, 4:] + 8*y[:, 3:-1] - 8*y[:, 1:-3] + y[:, 0:-4]
        ) / (12 * dt)

        # Forward edges
        d[:, 0] = (-25*y[:, 0] + 48*y[:, 1] - 36*y[:, 2] + 16*y[:, 3] - 3*y[:, 4]) / (12*dt)
        d[:, 1] = (-25*y[:, 1] + 48*y[:, 2] - 36*y[:, 3] + 16*y[:, 4] - 3*y[:, 5]) / (12*dt)

        # Backward edges
        n = y.shape[1]
        d[:, n-2] = (25*y[:, n-2] - 48*y[:, n-3] + 36*y[:, n-4] - 16*y[:, n-5] + 3*y[:, n-6]) / (12*dt)
        d[:, n-1] = (25*y[:, n-1] - 48*y[:, n-2] + 36*y[:, n-3] - 16*y[:, n-4] + 3*y[:, n-5]) / (12*dt)

        dSdt[ch] = xr.DataArray(
            d,
            dims=('event_index', 'time'),
            coords={
                'event_index': S['event_index'][:num_records],
                'time': S['time']
            },
            name=f'd{ch}_dt'
        )

    return xr.Dataset(dSdt)

