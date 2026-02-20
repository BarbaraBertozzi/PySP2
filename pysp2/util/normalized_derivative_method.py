import numpy as np
import xarray as xr

def central_difference(S, num_records=None, normalize=True):

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
    normalize: bool
        If True, normalize the derivative by the scattering signal 
        S(t) to get (1/S) * dS/dt.

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

        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                d = np.where(y != 0, d / y, 0)  
        else:
            d = d

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


def plot_normalized_derivative(ds, record_no, chn=0):
    """
    Plots the normalized derivative of the scattering signal for a given record_no and channel.

    Parameters
    ----------
    ds: xarray Dataset
        The dataset containing the normalized derivative to plot.
    record_no: int
        The record number to plot.
    chn: int
        The channel number to plot (0 or 4).
    Returns
    -------
    ax: matplotlib Axes
        The axes object containing the plot.
    """
    import matplotlib.pyplot as plt

    if chn not in [0, 4]:
        raise ValueError("Channel number must be 0 or 4.")
    
    spectra = ds.isel(event_index=record_no)
    time = spectra['time'].values
    inp_data = {}
    inp_data['time'] = xr.DataArray(np.array(time[np.newaxis]),
                                    dims=['time'])
    inp_data['Data_ch' + str(chn)] = xr.DataArray(
        spectra['Data_ch' + str(chn)].values[np.newaxis, :],
        dims=['time', 'bins'])
    inp_data = xr.Dataset(inp_data)
    bins = np.linspace(0, 100, 100)

    ch_name = f'Data_ch{chn}'
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    inp_data[ch_name].plot(ax=ax)
    ax.set_title(f'Normalized Derivative of Scattering Signal - Channel {chn} Record {record_no}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Derivative')
    plt.grid()

    return ax
