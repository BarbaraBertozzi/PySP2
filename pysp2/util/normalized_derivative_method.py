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
    dim: Optional[str] = None,
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
    dim : str, optional
        Name of the time dimension. If None, inferred as the only shared dimension.
    tau_grid : array-like, optional
        If provided, uses this global grid for tau for every k (in same units as t coordinate).
        If None, a per-k grid is constructed around the local time range.
    kend : int, optional
        Max starting index k to evaluate. If None, kend = N - p.
    config : MLEConfig, optional
        Required parameters from Appendix A. If None, raises ValueError.

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
        raise ValueError("config must be provided (contains h, sigma_bar, delta_sigma, A1..A3).")

    # Align inputs
    S, norm_deriv = xr.align(S, norm_deriv, join="inner")

    # Infer dimension
    if dim is None:
        common_dims = list(set(S.dims).intersection(norm_deriv.dims))
        if len(common_dims) != 1:
            raise ValueError(f"Could not infer dim uniquely. Provide dim. Common dims: {common_dims}")
        dim = common_dims[0]

    if dim not in S.dims:
        raise ValueError(f"dim='{dim}' not found in S.dims={S.dims}")
    if dim not in norm_deriv.dims:
        raise ValueError(f"dim='{dim}' not found in norm_deriv.dims={norm_deriv.dims}")

    # Extract numpy arrays
    y = np.asarray(norm_deriv.transpose(dim).data, dtype=float)
    s = np.asarray(S.transpose(dim).data, dtype=float)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("This function currently supports 1D DataArrays along `dim`.")
    if y.shape != s.shape:
        raise ValueError(f"S and norm_deriv must have same length; got {s.shape} vs {y.shape}.")

    n = y.size
    if p < 2 or p > n:
        raise ValueError(f"p must be in [2, {n}] but got {p}.")

    if k_end is None:
        k_end = n - p
    if k_end < 0 or k_end > n - p:
        raise ValueError(f"k_end must be in [0, {n-p}] but got {k_end}.")

    # Time coordinate (must be numeric)
    if dim not in S.coords:
        t = np.arange(n, dtype=float)
        t_units = None
    else:
        t = np.asarray(S[dim].data, dtype=float)
        t_units = S[dim].attrs.get("units")

    # Constants from Appendix A
    h = float(config.h)
    sigma_bar = float(config.sigma_bar)
    delta_sigma = float(config.delta_sigma)
    A1, A2, A3 = float(config.A1), float(config.A2), float(config.A3)

    if h <= 0 or sigma_bar <= 0:
        raise ValueError("h and sigma_bar must be positive.")
    if delta_sigma < 0:
        raise ValueError("delta_sigma must be >= 0.")

    # Eq. (A.7): Af_d = sqrt(130)/12
    Af_d = np.sqrt(130.0) / 12.0

    # Eq. (A.6): deltaS_i
    # deltaS_i = sqrt(A1^2 + A2^2 S_i + A3^2 S_i^2)
    # Note: if S can be negative in your data, consider preprocessing upstream (paper doesn't specify clipping).
    deltaS = np.sqrt(A1 * A1 + (A2 * A2) * s + (A3 * A3) * (s * s))

    # Random variance term in Var[y_i], Eq. (A.10b) using Eq. (A.7)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_rand = (Af_d * Af_d) / (h * h) * (deltaS * deltaS) / (s * s)

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

    tau_hat = np.full(k_end + 1, np.nan, dtype=float)

    def _logL_for_tau(yk: np.ndarray, tk: np.ndarray, var_rand_k: np.ndarray, tau: float) -> float:
        """
        Compute log L_k(tau) for one subset (k) and one candidate tau, per Eq. (A.9).
        Uses Cholesky factorization for stability.
        """
        # Mean vector (Eq. A.4)
        ybar = -(tk - tau) / (sigma_bar * sigma_bar)

        # Covariance matrix (Eqs. A.10a,b)
        dt = (tk - tau).reshape(-1, 1)  # (p,1)
        sys_pref = 4.0 * (delta_sigma * delta_sigma) / (sigma_bar ** 6)
        Sigma = sys_pref * (dt @ dt.T)
        Sigma[np.diag_indices_from(Sigma)] += var_rand_k

        r = yk - ybar

        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            return -np.inf

        # d2 = r^T Sigma^{-1} r via triangular solves
        z = np.linalg.solve(L, r)
        d2 = float(z.T @ z)

        # log|Sigma|
        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        p_local = yk.size
        return float(-0.5 * (p_local * np.log(2.0 * np.pi) + logdet + d2))

    for k in range(k_end + 1):
        yk = y[k : k + p]
        tk = t[k : k + p]
        var_rand_k = var_rand[k : k + p]

        # Basic sanity checks
        if not (np.all(np.isfinite(yk)) and np.all(np.isfinite(tk)) and np.all(np.isfinite(var_rand_k))):
            continue
        if np.any(var_rand_k <= 0):
            continue

        # Per-k grid if not provided
        if tau_grid_np is None:
            span = float(tk[-1] - tk[0])
            margin = config.grid_margin * (span + h)
            grid = np.linspace(tk[0] - margin, tk[-1] + margin, config.grid_size)
        else:
            grid = tau_grid_np

        best_ll = -np.inf
        best_tau = np.nan

        for tau_cand in grid:
            ll = _logL_for_tau(yk, tk, var_rand_k, float(tau_cand))
            if ll > best_ll:
                best_ll = ll
                best_tau = float(tau_cand)

        if np.isfinite(best_ll):
            tau_hat[k] = best_tau

    # Return as DataArray
    k_coord = xr.DataArray(np.arange(k_end + 1), dims=("k",), name="k")
    out = xr.DataArray(tau_hat, dims=("k",), coords={"k": k_coord}, name="tau_hat")
    out.attrs["long_name"] = "MLE estimate of tau for each k-subset"
    if t_units:
        out.attrs["units"] = t_units

    return out
