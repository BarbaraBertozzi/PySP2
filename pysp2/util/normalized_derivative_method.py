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
    # Instrument / calibration parameters used in Appendix A
    h: float  # sampling interval (same units as t)
    sigma_bar: float  # average Gaussian width (\\bar{sigma})
    delta_sigma: float  # std dev of width fluctuation (\\delta sigma)
    A1: float  # noise params for delta S_i (Eq. A.6)
    A2: float
    A3: float

    # Grid-search controls
    grid_size: int = 401
    grid_margin: float = 0.5  # margin factor relative to local time span


def mle_tau_moteki_kondo(
    S: xr.DataArray,
    norm_deriv: xr.DataArray,
    p: int,
    *,
    event_index: Optional[int] = None,
    event_dim: str = "event_index",
    S_sample_dim: Optional[str] = None,
    y_sample_dim: Optional[str] = None,
    tau_grid: Optional[Union[np.ndarray, xr.DataArray]] = None,
    k_end: Optional[int] = None,
    config: Optional[MLEConfig] = None,
) -> xr.DataArray:
    """
    Maximum-likelihood estimation of tau for each k-subset using Moteki & Kondo (2008) Appendix A.

    For each k = 0..k_end, finds:
      - tau_hat(k) = argmax_tau L_k(tau)
            where L_k is the multivariate normal likelihood (Eq. A.9) with:
            - mean ybar_i(tau) = -(t_i - tau)/sigma_bar^2 (Eq. A.4)
            - covariance Sigma_k(tau) from Eqs. (A.10a,b), using S-dependent noise (A.6,A.7)

      Implements:
      - Likelihood L_k(tau) as multivariate normal (Eq. A.9)
      - MLE of tau by grid search (Appendix A.5)

    Parameters
    ----------
    S : xr.DataArray
        Scattering signal S(t) along a single time-like dimension.
    norm_deriv : xr.DataArray
        Normalized derivative y(t) = S'(t)/S(t), same dimension/coords as S (or alignable).
    p : int
        Sub-array length (number of consecutive points) used for each k.
    event_index : int, optional
        If given, compute tau_hat only for this one event and return tau_hat(k).
        If None, compute tau_hat for all events and return tau_hat(event_index, k).
    event_dim : str, default "event_index"
        Name of the event dimension.
    S_sample_dim : str, optional
        Sample dimension in S. If None, inferred as the non-event dimension.
    y_sample_dim : str, optional
        Sample dimension in norm_deriv. If None, inferred as the non-event dimension.
    tau_grid : 1D array-like, optional
        Global tau grid to use for all subsets. If None, a per-k grid is constructed.
    k_end : int, optional
        Largest starting k. If None, uses n_samples - p.
    config : MLEConfig
        Calibration / noise / grid settings.

    Returns
    -------
    xr.Dataset with variables:
      - tau_hat(k): MLE estimate of tau for each k

    Notes
    -----
    - This function estimates tau for every k. Moteki & Kondo then choose k_best as the k
      with minimum d^2(k) (Appendix A.5), and apply a chi-square test on d^2(k_best).
    - For numerical stability, likelihood is computed via Cholesky factorization of Sigma_k.
    - Currently supports 1D DataArrays along `dim`.
    """
    if config is None:
        raise ValueError("config must be provided.")

    if event_dim not in S.dims:
        raise ValueError(f"{event_dim!r} not found in S.dims={S.dims}")
    if event_dim not in norm_deriv.dims:
        raise ValueError(f"{event_dim!r} not found in norm_deriv.dims={norm_deriv.dims}")

    # Infer sample dimensions
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

    # Standardize dimension names internally
    S_std = S.rename({S_sample_dim: "sample"})
    y_std = norm_deriv.rename({y_sample_dim: "sample"})

    # Align on event and sample positions
    S_std, y_std = xr.align(S_std, y_std, join="inner")

    if S_std.sizes["sample"] != y_std.sizes["sample"]:
        raise ValueError("S and norm_deriv must have the same number of samples per event.")

    n_events = S_std.sizes[event_dim]
    n_samples = S_std.sizes["sample"]

    if p < 2 or p > n_samples:
        raise ValueError(f"p must be in [2, {n_samples}], got {p}")

    if k_end is None:
        k_end = n_samples - p
    if k_end < 0 or k_end > n_samples - p:
        raise ValueError(f"k_end must be in [0, {n_samples - p}], got {k_end}")

    # Use sample index as t-axis
    t = np.arange(n_samples, dtype=float)

    # Optional global tau grid
    if tau_grid is not None:
        tau_grid_np = np.asarray(
            tau_grid.data if isinstance(tau_grid, xr.DataArray) else tau_grid,
            dtype=float,
        )
        if tau_grid_np.ndim != 1:
            raise ValueError("tau_grid must be 1D.")
    else:
        tau_grid_np = None

    # Constants from Appendix A
    h = float(config.h)
    sigma_bar = float(config.sigma_bar)
    delta_sigma = float(config.delta_sigma)
    A1, A2, A3 = float(config.A1), float(config.A2), float(config.A3)

    if h <= 0:
        raise ValueError("config.h must be positive.")
    if sigma_bar <= 0:
        raise ValueError("config.sigma_bar must be positive.")
    if delta_sigma < 0:
        raise ValueError("config.delta_sigma must be >= 0.")

    Af_d = np.sqrt(130.0) / 12.0  # Eq. (A.7)

    def _logL_for_tau(yk: np.ndarray, sk: np.ndarray, tk: np.ndarray, tau: float) -> float:
        """
        Compute log L_k(tau) for one subset (k) and one candidate tau, per Eq. (A.9).
        Uses Cholesky factorization for stability.
        """

        # Eq. (A.4)
        ybar = -(tk - tau) / (sigma_bar * sigma_bar)

        # Eq. (A.6)
        deltaS = np.sqrt(A1 * A1 + (A2 * A2) * sk + (A3 * A3) * (sk * sk))

        # Eq. (A.10b) random term, using Eq. (A.7)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_rand_k = (Af_d * Af_d) / (h * h) * (deltaS * deltaS) / (sk * sk)

        if not np.all(np.isfinite(var_rand_k)):
            return -np.inf
        if np.any(var_rand_k <= 0):
            return -np.inf

        # Eqs. (A.10a,b)
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
        """Return tau_hat(k) for a single event."""
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

    # Compute one event only
    if event_index is not None:
        s_event = np.asarray(S_std.sel({event_dim: event_index}).data, dtype=float)
        y_event = np.asarray(y_std.sel({event_dim: event_index}).data, dtype=float)

        tau_hat_1d = _tau_hat_for_one_event(s_event, y_event)

        out = xr.DataArray(
            tau_hat_1d,
            dims=("k",),
            coords={"k": np.arange(k_end + 1)},
            name="tau_hat",
        )
        out.attrs["long_name"] = f"MLE tau_hat(k) for {event_dim}={event_index}"
        out.attrs["units"] = "sample_index"
        return out

    # Compute all events
    tau_hat_all = np.full((n_events, k_end + 1), np.nan, dtype=float)

    event_vals = (
        S_std[event_dim].values
        if event_dim in S_std.coords
        else np.arange(n_events)
    )

    for i in range(n_events):
        s_event = np.asarray(S_std.isel({event_dim: i}).data, dtype=float)
        y_event = np.asarray(y_std.isel({event_dim: i}).data, dtype=float)
        tau_hat_all[i, :] = _tau_hat_for_one_event(s_event, y_event)

    out = xr.DataArray(
        tau_hat_all,
        dims=(event_dim, "k"),
        coords={
            event_dim: event_vals,
            "k": np.arange(k_end + 1),
        },
        name="tau_hat",
    )
    out.attrs["long_name"] = "MLE tau_hat(event_index, k)"
    out.attrs["units"] = "sample_index"
    return out
