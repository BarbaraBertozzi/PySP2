from __future__ import annotations

import numpy as np
import xarray as xr

from dataclasses import dataclass
from typing import Optional, Union


def central_difference(S, num_records=None, normalize=True, baseline_to_zero=True):

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
    baseline_to_zero: bool
        If True, shift each record's minimum to zero before differentiation.

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

        # Baseline shift: make each record's minimum be 0
        if baseline_to_zero:
            y_min = np.nanmin(y, axis=1, keepdims=True)   # shape (n_records, 1)
            y = y - y_min
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

    bins = np.arange(0, 0.00004-0.3e-6, 0.4e-6)  # 0 to 0.0004 microseconds in steps of 0.4e-6 seconds
    bins = bins*1e6  # convert to microseconds for plotting

    ch_name = f'Data_ch{chn}'
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Plot using bins for x-axis
    ax.plot(bins, spectra['Data_ch' + str(chn)].values, label=ch_name)
    ax.set_xlim([bins[0], bins[-1]])
    ax.set_title(f'Normalized Derivative of Scattering Signal - Channel {chn} Record {record_no}')
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_ylabel('Normalized Derivative')
    plt.grid()
    ax.legend()

    return ax


@dataclass(frozen=True)
class MLEConfig:
    h: float
    sigma_bar: float
    delta_sigma: float
    A1: float
    A2: float
    A3: float
    grid_size: int = 401
    grid_margin: float = 0.5


def mle_tau_moteki_kondo(
    S: Union[xr.DataArray, xr.Dataset],
    norm_deriv: Union[xr.DataArray, xr.Dataset],
    p: int,
    *,
    data_var: Optional[str] = None,
    event_index: Optional[int] = None,
    event_dim: str = "event_index",
    S_sample_dim: Optional[str] = None,
    y_sample_dim: Optional[str] = None,
    tau_grid: Optional[Union[np.ndarray, xr.DataArray]] = None,
    k_end: Optional[int] = None,
    config: Optional[MLEConfig] = None,
) -> xr.DataArray:
    """
    Estimate tau_hat using the Moteki & Kondo grid-search MLE.

    Parameters
    ----------
    S : xr.DataArray or xr.Dataset
        Scattering signal.
    norm_deriv : xr.DataArray or xr.Dataset
        Normalized derivative.
    p : int
        Number of consecutive points in each k-subset.
    data_var : str, optional
        Variable to select when S and/or norm_deriv are Datasets.
        Required if a Dataset contains multiple variables and no unique choice exists.
    event_index : int, optional
        If given, return tau_hat(k) for one event.
        If None, return tau_hat(event_index, k) for all events.
    event_dim : str
        Name of event dimension.
    S_sample_dim : str, optional
        Sample dimension in S.
    y_sample_dim : str, optional
        Sample dimension in norm_deriv.
    tau_grid : 1D array-like, optional
        Global tau grid for all subsets.
    k_end : int, optional
        Largest starting k.
    config : MLEConfig
        Calibration / noise / grid settings.
    """
    if config is None:
        raise ValueError("config must be provided.")

    def _to_dataarray(obj: Union[xr.DataArray, xr.Dataset], name: str) -> xr.DataArray:
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if data_var is not None:
                if data_var not in obj.data_vars:
                    raise ValueError(
                        f"{data_var!r} not found in {name}.data_vars={list(obj.data_vars)}"
                    )
                return obj[data_var]
            if len(obj.data_vars) == 1:
                only_var = next(iter(obj.data_vars))
                return obj[only_var]
            raise ValueError(
                f"{name} is a Dataset with multiple variables. "
                f"Provide data_var. Available: {list(obj.data_vars)}"
            )
        raise TypeError(f"{name} must be an xarray DataArray or Dataset.")

    S = _to_dataarray(S, "S")
    norm_deriv = _to_dataarray(norm_deriv, "norm_deriv")

    if event_dim not in S.dims:
        raise ValueError(f"{event_dim!r} not found in S.dims={S.dims}")
    if event_dim not in norm_deriv.dims:
        raise ValueError(f"{event_dim!r} not found in norm_deriv.dims={norm_deriv.dims}")

    if S_sample_dim is None:
        s_non_event_dims = [d for d in S.dims if d != event_dim]
        if len(s_non_event_dims) != 1:
            raise ValueError(
                f"Could not infer S sample dim. Non-event dims in S: {s_non_event_dims}"
            )
        S_sample_dim = s_non_event_dims[0]

    if y_sample_dim is None:
        y_non_event_dims = [d for d in norm_deriv.dims if d != event_dim]
        if len(y_non_event_dims) != 1:
            raise ValueError(
                f"Could not infer norm_deriv sample dim. Non-event dims in norm_deriv: {y_non_event_dims}"
            )
        y_sample_dim = y_non_event_dims[0]

    S_std = S.rename({S_sample_dim: "sample"})
    y_std = norm_deriv.rename({y_sample_dim: "sample"})

    S_std, y_std = xr.align(S_std, y_std, join="inner")

    n_events = S_std.sizes[event_dim]
    n_samples = S_std.sizes["sample"]

    if p < 2 or p > n_samples:
        raise ValueError(f"p must be in [2, {n_samples}], got {p}")

    if k_end is None:
        k_end = n_samples - p
    if k_end < 0 or k_end > n_samples - p:
        raise ValueError(f"k_end must be in [0, {n_samples - p}], got {k_end}")

    #t = np.arange(n_samples, dtype=float)

    if tau_grid is not None:
        tau_grid_np = np.asarray(
            tau_grid.data if isinstance(tau_grid, xr.DataArray) else tau_grid,
            dtype=float,
        )
        if tau_grid_np.ndim != 1:
            raise ValueError("tau_grid must be 1D.")
    else:
        tau_grid_np = None

    h = float(config.h)
    sigma_bar = float(config.sigma_bar)
    delta_sigma = float(config.delta_sigma)
    A1, A2, A3 = float(config.A1), float(config.A2), float(config.A3)
    
    t = np.arange(n_samples) * h

    if h <= 0:
        raise ValueError("config.h must be positive.")
    if sigma_bar <= 0:
        raise ValueError("config.sigma_bar must be positive.")
    if delta_sigma < 0:
        raise ValueError("config.delta_sigma must be >= 0.")

    Af_d = np.sqrt(130.0) / 12.0

    def _logL_for_tau(yk: np.ndarray, sk: np.ndarray, tk: np.ndarray, tau: float) -> float:
        ybar = -(tk - tau) / (sigma_bar * sigma_bar)

        deltaS = np.sqrt(A1 * A1 + (A2 * A2) * sk + (A3 * A3) * (sk * sk))

        with np.errstate(divide="ignore", invalid="ignore"):
            var_rand_k = (Af_d * Af_d) / (h * h) * (deltaS * deltaS) / (sk * sk)

        if not np.all(np.isfinite(var_rand_k)):
            return -np.inf
        if np.any(var_rand_k <= 0):
            return -np.inf

        dt = (tk - tau).reshape(-1, 1)
        sys_pref = 4.0 * (delta_sigma * delta_sigma) / (sigma_bar ** 6)
        Sigma = sys_pref * (dt @ dt.T)
        Sigma[np.diag_indices_from(Sigma)] += var_rand_k

        r = yk - ybar

        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            return -np.inf

        z = np.linalg.solve(L, r)
        d2 = float(z.T @ z)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        p_local = yk.size
        return float(-0.5 * (p_local * np.log(2.0 * np.pi) + logdet + d2))

    def _tau_hat_for_one_event(s_event: np.ndarray, y_event: np.ndarray) -> np.ndarray:
        tau_hat = np.full(k_end + 1, np.nan, dtype=float)

        if not (np.all(np.isfinite(s_event)) and np.all(np.isfinite(y_event))):
            return tau_hat

        for k in range(k_end + 1):
            yk = y_event[k:k + p]
            sk = s_event[k:k + p]
            tk = t[k:k + p]

            if not (np.all(np.isfinite(yk)) and np.all(np.isfinite(sk))):
                continue

            if tau_grid_np is None:
                span = float(tk[-1] - tk[0])
                margin = config.grid_margin * (span + h)
                grid = np.linspace(tk[0] - margin, tk[-1] + margin, config.grid_size)
            else:
                grid = tau_grid_np

            best_ll = -np.inf
            best_tau = np.nan

            for tau_cand in grid:
                ll = _logL_for_tau(yk, sk, tk, float(tau_cand))
                if ll > best_ll:
                    best_ll = ll
                    best_tau = float(tau_cand)

            if np.isfinite(best_ll):
                tau_hat[k] = best_tau

        return tau_hat

    if event_index is not None:
        s_event = np.asarray(S_std.sel({event_dim: event_index}).values, dtype=float)
        y_event = np.asarray(y_std.sel({event_dim: event_index}).values, dtype=float)

        tau_hat_1d = _tau_hat_for_one_event(s_event, y_event)

        return xr.DataArray(
            tau_hat_1d,
            dims=("k",),
            coords={"k": np.arange(k_end + 1)},
            name="tau_hat",
            attrs={"long_name": f"MLE tau_hat(k) for {event_dim}={event_index}",
                   "units": "sample_index"},
        )

    tau_hat_all = np.full((n_events, k_end + 1), np.nan, dtype=float)

    event_vals = (
        S_std[event_dim].values
        if event_dim in S_std.coords
        else np.arange(n_events)
    )

    for i in range(n_events):
        s_event = np.asarray(S_std.isel({event_dim: i}).values, dtype=float)
        y_event = np.asarray(y_std.isel({event_dim: i}).values, dtype=float)
        tau_hat_all[i, :] = _tau_hat_for_one_event(s_event, y_event)

    return xr.DataArray(
        tau_hat_all,
        dims=(event_dim, "k"),
        coords={event_dim: event_vals, "k": np.arange(k_end + 1)},
        name="tau_hat",
        attrs={"long_name": "MLE tau_hat(event_index, k)", "units": "sample_index"},
    )
