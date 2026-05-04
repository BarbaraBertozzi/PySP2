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
    """
    Configuration parameters for Maximum Likelihood Estimation (MLE)
    of the Moteki & Kondo normalized derivative method.

    Parameters
    ----------
    h : float
        Time resolution of the scattering signal (instrument specification).
    sigma_bar : float
        Mean value of the noise standard deviation (measured value).
    delta_sigma : float
        Increment for the noise standard deviation (measured value).
    A1 : float
        Coefficient for the first term in the model (determined experimentally).
    A2 : float
        Coefficient for the second term in the model (determined experimentally).
    A3 : float
        Coefficient for the third term in the model (determined experimentally).
    grid_size : int, default=401
        Number of grid points for tau estimation.
    grid_margin : float, default=0.5
        Margin for the tau grid as a fraction of the range.
    """
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
    ch: Optional[str] = None,
    event_index: int,
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
    ch : str, optional
        Variable to select when S and/or norm_deriv are Datasets.
        Required if a Dataset contains multiple variables and no unique choice exists.
    event_index : int
        Event index to select. This function returns tau_hat(k) for one event only.
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
        """
        Accept either a DataArray or Dataset.
        If a Dataset is provided, select the variable named `ch`.
        """
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            if ch is not None:
                # Use the user input channel.
                if ch not in obj.data_vars:
                    raise ValueError(
                        f"{ch!r} not found in {name}.data_vars={list(obj.data_vars)}"
                    )
                return obj[ch]
            if len(obj.data_vars) == 1:
                only_var = next(iter(obj.data_vars))
                return obj[only_var]
            raise ValueError(
                f"{name} is a Dataset with multiple variables. "
                f"Provide ch. Available: {list(obj.data_vars)}"
            )
        raise TypeError(f"{name} must be an xarray DataArray or Dataset.")

    # Convert datasets to the selected DataArrays.
    S = _to_dataarray(S, "S")
    norm_deriv = _to_dataarray(norm_deriv, "norm_deriv")

    # The method requires one event axis and one sample axis.
    if event_dim not in S.dims:
        raise ValueError(f"{event_dim!r} not found in S.dims={S.dims}")
    if event_dim not in norm_deriv.dims:
        raise ValueError(f"{event_dim!r} not found in norm_deriv.dims={norm_deriv.dims}")

    # Infer the sample dimension if the user did not specify it.
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

    # Rename the sample dimensions to a common internal name.
    S_std = S.rename({S_sample_dim: "sample"})
    y_std = norm_deriv.rename({y_sample_dim: "sample"})

    # Align the arrays so the same event/sample positions are used in both inputs.
    S_std, y_std = xr.align(S_std, y_std, join="inner")

    if event_index < 0 or event_index >= S_std.sizes[event_dim]:
        raise ValueError(
            f"event_index must be in [0, {S_std.sizes[event_dim] - 1}], got {event_index}"
        )

    n_samples = S_std.sizes["sample"]

    if p < 2 or p > n_samples:
        raise ValueError(f"p must be in [2, {n_samples}], got {p}")

    if k_end is None:
        k_end = n_samples - p
    if k_end < 0 or k_end > n_samples - p:
        raise ValueError(f"k_end must be in [0, {n_samples - p}], got {k_end}")

    # Optional tau grid for the 1D grid search in tau.
    # Moteki & Kondo determine tau numerically by maximizing L_k(tau).
    if tau_grid is not None:
        tau_grid_np = np.asarray(
            tau_grid.data if isinstance(tau_grid, xr.DataArray) else tau_grid,
            dtype=float,
        )
        if tau_grid_np.ndim != 1:
            raise ValueError("tau_grid must be 1D.")
    else:
        tau_grid_np = None

    # Parameters from Appendix A.
    h = float(config.h)
    sigma_bar = float(config.sigma_bar)
    delta_sigma = float(config.delta_sigma)
    A1, A2, A3 = float(config.A1), float(config.A2), float(config.A3)

    # Time axis used in the fit.
    # Here we use physical time spacing h so tk is in seconds (or whatever unit h uses).
    # This must match sigma_bar and delta_sigma units.
    t = np.arange(n_samples) * h

    if h <= 0:
        raise ValueError("config.h must be positive.")
    if sigma_bar <= 0:
        raise ValueError("config.sigma_bar must be positive.")
    if delta_sigma < 0:
        raise ValueError("config.delta_sigma must be >= 0.")

    # Eq. (A.7): finite-difference amplification factor for the derivative noise.
    Af_d = np.sqrt(130.0) / 12.0

    def _logL_for_tau(yk: np.ndarray, sk: np.ndarray, tk: np.ndarray, tau: float) -> float:
        """
        Log-likelihood for one k-subset and one candidate tau.

        Mean model:
            ybar_i(tau) = -(t_i - tau) / sigma_bar^2      [Eq. (A.4)]
        where y_i = S'_i / S_i.

        Covariance:
            Cov[y_i, y_j] = 4 / sigma_bar^6 * (t_i - tau)(t_j - tau) * (delta_sigma)^2   [Eq. (A.10a)]
            Var[y_i]      = 4 / sigma_bar^6 * (t_i - tau)^2 * (delta_sigma)^2
                            + (Af_d^2 / h^2) * (1/S_i^2) * (delta S_i)^2                  [Eq. (A.10b)]
        with
            delta S_i = sqrt(A1^2 + A2^2 S_i + A3^2 S_i^2)                               [Eq. (A.6)]
        and
            (delta y_i)_ran = Af_d * (1/h) * (1/S_i) * delta S_i                          [Eq. (A.7)]

        The full likelihood is the multivariate Gaussian in Eq. (A.9).
        """
        # Mean vector of the normalized derivative under the Gaussian beam model.
        # This is the line I'/I = -(t - tau)/sigma^2 [Eq. (5)] used as the mean [Eq. (A.4)].
        ybar = -(tk - tau) / (sigma_bar * sigma_bar)

        # Signal-noise amplitude from Appendix A [Eq. (A.6)].
        deltaS = np.sqrt(A1 * A1 + (A2 * A2) * sk + (A3 * A3) * (sk * sk))

        # Random variance of y = S'/S from finite-difference error propagation [Eq. (A.7)].
        with np.errstate(divide="ignore", invalid="ignore"):
            var_rand_k = (Af_d * Af_d) / (h * h) * (deltaS * deltaS) / (sk * sk)

        # If any term is non-finite, this tau candidate is unusable.
        if not np.all(np.isfinite(var_rand_k)):
            return -np.inf
        if np.any(var_rand_k <= 0):
            return -np.inf

        # Systematic covariance from particle-by-particle fluctuations in sigma [Eq. (A.10a)].
        dt = (tk - tau).reshape(-1, 1)
        sys_pref = 4.0 * (delta_sigma * delta_sigma) / (sigma_bar ** 6)
        Sigma = sys_pref * (dt @ dt.T)
        # Add the diagonal random variance term [Eq. (A.10b)].
        Sigma[np.diag_indices_from(Sigma)] += var_rand_k

        # Residual vector y - ybar.
        r = yk - ybar

        # Use Cholesky factorization for numerical stability when evaluating Eq. (A.9).
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            return -np.inf

        # Compute statistical distance.
        # d^2 = (y - ybar)^T Sigma^{-1} (y - ybar) [Eq. (A.11)]
        z = np.linalg.solve(L, r)
        d2 = float(z.T @ z)
        # log |Sigma| from the Cholesky factor.
        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        # Multivariate normal log-likelihood [Eq. (A.9)].
        p_local = yk.size
        return float(-0.5 * (p_local * np.log(2.0 * np.pi) + logdet + d2))

    def _tau_hat_for_one_event(s_event: np.ndarray, y_event: np.ndarray) -> np.ndarray:
        """
        For one event, scan all k-subsets of length p and return tau_hat(k).
        """
        tau_hat = np.full(k_end + 1, np.nan, dtype=float)

        # Skip events with missing values.
        if not (np.all(np.isfinite(s_event)) and np.all(np.isfinite(y_event))):
            return tau_hat

        for k in range(k_end + 1):
            # Consecutive p-point subset starting at k.
            # This is the subset over which Moteki & Kondo search for the leading-edge
            # segment that best matches I'/I [Appendix A.5].
            yk = y_event[k : k + p]
            sk = s_event[k : k + p]
            tk = t[k : k + p]

            if not (np.all(np.isfinite(yk)) and np.all(np.isfinite(sk))):
                continue

            # If the user did not supply a global tau grid, build a local grid for this k.
            if tau_grid_np is None:
                span = float(tk[-1] - tk[0])
                margin = config.grid_margin * (span + h)
                grid = np.linspace(tk[0] - margin, tk[-1] + margin, config.grid_size)
            else:
                grid = tau_grid_np

            # Grid-search maximization of L_k(tau) [Appendix A.5].
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

    # Select the requested event only, and return tau_hat(k) for that one event.
    s_event = np.asarray(S_std.sel({event_dim: event_index}).values, dtype=float)
    y_event = np.asarray(y_std.sel({event_dim: event_index}).values, dtype=float)

    tau_hat_1d = _tau_hat_for_one_event(s_event, y_event)

    return xr.DataArray(
        tau_hat_1d,
        dims=("k",),
        coords={"k": np.arange(k_end + 1)},
        name="tau_hat",
        attrs={
            "long_name": f"MLE tau_hat(k) for {event_dim}={event_index}",
            "units": "sample_index_or_time_units_of_t",
        },
    )
