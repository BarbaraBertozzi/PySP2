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


def plot_normalized_derivative(S, ds, record_no, chn=0, plot_scattering_signal=False):
    """
    Plots the normalized derivative of the scattering signal for a given record_no and channel.

    Parameters
    ----------
    S: xarray Dataset
        The original scattering signal dataset, used for optional overlay.
    ds: xarray Dataset
        The dataset containing the normalized derivative to plot.
    record_no: int
        The record number to plot.
    chn: int
        The channel number to plot (0 or 4).
    plot_scattering_signal: bool
        If True, overlay the original scattering signal on the plot.
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
    #inp_data['Data_ch' + str(chn)] = xr.DataArray(
    #    spectra['Data_ch' + str(chn)].values[np.newaxis, :],
    #    dims=['time', 'bins'])
    #inp_data = xr.Dataset(inp_data)

    bins = np.arange(0, 0.00004-0.3e-6, 0.4e-6)  # 0 to 0.0004 microseconds in steps of 0.4e-6 seconds
    bins = bins*1e6  # convert to microseconds for plotting

    ch_name = f'Data_ch{chn}'
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
 # --- Primary axis: normalized derivative ---
    line1, = ax.plot(
        bins,
        spectra[ch_name].values,
        label=f'{ch_name} (Normalized dS/dt)'
    )

    ax.set_xlim([bins[0], bins[-1]])
    ax.set_xlabel('Time ($\\mu$s)')
    ax.set_ylabel('Normalized Derivative')

    # --- Secondary axis: scattering signal ---
    if plot_scattering_signal:
        ax2 = ax.twinx()
        y = S[ch_name].isel(event_index=record_no).values
        y = y - np.nanmin(y)  # baseline shift

        line2, = ax2.plot(
            bins,
            y,
            color = 'black',
            linestyle='--',
            label=f'{ch_name} (Scattering Signal)'
        )
        ax2.set_ylabel('Scattering Signal (baseline shifted)')

        # Combine legends
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels)
    else:
        ax.legend()
    ax.set_title(
        f'Normalized Derivative of Scattering Signal - Channel {chn} Record {record_no}'
    )
    ax.grid()

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

def _to_dataarray(
    obj: Union[xr.DataArray, xr.Dataset],
    name: str,
    ch: Optional[str] = None,
) -> xr.DataArray:
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


def _moteki_kondo_subset_statistics(
    yk: np.ndarray,
    sk: np.ndarray,
    tk: np.ndarray,
    tau: float,
    *,
    h: float,
    sigma_bar: float,
    delta_sigma: float,
    A1: float,
    A2: float,
    A3: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Shared Moteki & Kondo subset statistics.

    Returns
    -------
    ybar : np.ndarray
        Mean vector [Eq. (A.4)].
    Sigma : np.ndarray
        Covariance matrix [Eqs. (A.10a), (A.10b)].
    L : np.ndarray or None
        Cholesky factor of Sigma, if Sigma is positive definite.
    d2 : float or None
        Statistical distance squared [Eq. (A.11)].
    """
    # Mean vector of the normalized derivative under the Gaussian beam model.
    # Eq. (A.4): ybar_i(tau) = -(t_i - tau) / sigma_bar^2
    ybar = -(tk - tau) / (sigma_bar * sigma_bar)

    # Signal-noise amplitude from Appendix A.
    # Eq. (A.6): deltaS_i = sqrt(A1^2 + A2^2 S_i + A3^2 S_i^2)
    deltaS = np.sqrt(A1 * A1 + (A2 * A2) * sk + (A3 * A3) * (sk * sk))

    # Random variance in y = S'/S from finite-difference error propagation.
    # Eq. (A.7): (delta y_i)_ran = Af_d * (1/h) * (1/S_i) * deltaS_i
    Af_d = np.sqrt(130.0) / 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        var_rand_k = (Af_d * Af_d) / (h * h) * (deltaS * deltaS) / (sk * sk)

    if not np.all(np.isfinite(var_rand_k)):
        return None, None, None, None
    if np.any(var_rand_k <= 0):
        return None, None, None, None

    # Systematic covariance from particle-by-particle fluctuations in sigma.
    # Eq. (A.10a): Cov[y_i, y_j] = 4/sigma_bar^6 * (t_i - tau)(t_j - tau) * delta_sigma^2
    # Eq. (A.10b): Var[y_i] adds the random variance term above.
    dt = (tk - tau).reshape(-1, 1)
    sys_pref = 4.0 * (delta_sigma * delta_sigma) / (sigma_bar ** 6)
    Sigma = sys_pref * (dt @ dt.T)
    Sigma[np.diag_indices_from(Sigma)] += var_rand_k

    r = yk - ybar

    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return ybar, Sigma, None, None

    # Eq. (A.11): d^2 = (y - ybar)^T Sigma^{-1} (y - ybar)
    z = np.linalg.solve(L, r)
    d2 = float(z.T @ z)

    return ybar, Sigma, L, d2

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

    # Convert datasets to the selected DataArrays.
    S = _to_dataarray(S, "S", ch=ch)
    norm_deriv = _to_dataarray(norm_deriv, "norm_deriv", ch=ch)

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
    # Moteki & Kondo determine tau numerically by maximizing L_k(tau) [Appendix A.5].
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
    t = np.arange(n_samples, dtype=float) * h

    if h <= 0:
        raise ValueError("config.h must be positive.")
    if sigma_bar <= 0:
        raise ValueError("config.sigma_bar must be positive.")
    if delta_sigma < 0:
        raise ValueError("config.delta_sigma must be >= 0.")

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
                _, _, _, d2 = _moteki_kondo_subset_statistics(
                    yk,
                    sk,
                    tk,
                    float(tau_cand),
                    h=h,
                    sigma_bar=sigma_bar,
                    delta_sigma=delta_sigma,
                    A1=A1,
                    A2=A2,
                    A3=A3,
                )

                if d2 is None:
                    ll = -np.inf
                else:
                    # Reconstruct the log-likelihood from the same subset statistics.
                    # Eq. (A.9): log L = -1/2 * [p log(2pi) + log|Sigma| + d^2]
                    # Since _moteki_kondo_subset_statistics does not return log|Sigma|,
                    # we compute likelihood separately here for the grid-search step.
                    ybar, Sigma, L, d2 = _moteki_kondo_subset_statistics(
                        yk,
                        sk,
                        tk,
                        float(tau_cand),
                        h=h,
                        sigma_bar=sigma_bar,
                        delta_sigma=delta_sigma,
                        A1=A1,
                        A2=A2,
                        A3=A3,
                    )
                    if L is None or d2 is None:
                        ll = -np.inf
                    else:
                        logdet = 2.0 * np.sum(np.log(np.diag(L)))
                        p_local = yk.size
                        ll = float(-0.5 * (p_local * np.log(2.0 * np.pi) + logdet + d2))

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

def compute_d2_moteki_kondo(
    S: Union[xr.DataArray, xr.Dataset],
    norm_deriv: Union[xr.DataArray, xr.Dataset],
    tau_hat: Union[np.ndarray, xr.DataArray],
    p: int,
    *,
    ch: Optional[str] = None,
    event_index: int,
    event_dim: str = "event_index",
    S_sample_dim: Optional[str] = None,
    y_sample_dim: Optional[str] = None,
    k_end: Optional[int] = None,
    config: Optional[MLEConfig] = None,
) -> xr.DataArray:
    """
    Compute d^2(k) for one selected event using the Moteki & Kondo statistical distance.

    This is the quantity used to quantify how well each k-subset matches the expected
    I'/I line segment [Eq. (A.11)], and it is the same statistic used in Appendix A.5
    for judging candidate sub-arrays against the chi-square reference distribution.

    Parameters
    ----------
    S : xr.DataArray or xr.Dataset
        Scattering signal S(t).
    norm_deriv : xr.DataArray or xr.Dataset
        Normalized derivative S'(t)/S(t).
    tau_hat : 1D array-like or xr.DataArray
        tau_hat(k) for the selected event. Must have length k_end + 1.
    p : int
        Number of consecutive points in each k-subset.
    ch : str, optional
        Variable name to select when S and/or norm_deriv are Datasets.
    event_index : int
        Event index to evaluate.
    event_dim : str
        Name of the event dimension.
    S_sample_dim : str, optional
        Sample dimension in S.
    y_sample_dim : str, optional
        Sample dimension in norm_deriv.
    k_end : int, optional
        Largest starting k. If None, inferred from tau_hat length.
    config : MLEConfig
        Calibration / noise / grid settings.

    Returns
    -------
    xr.DataArray
        d^2(k) for the selected event, with dimension k.
    """
    if config is None:
        raise ValueError("config must be provided.")

    # Convert Datasets to DataArrays if needed.
    S = _to_dataarray(S, "S", ch=ch)
    norm_deriv = _to_dataarray(norm_deriv, "norm_deriv", ch=ch)

    # Require the event dimension.
    if event_dim not in S.dims:
        raise ValueError(f"{event_dim!r} not found in S.dims={S.dims}")
    if event_dim not in norm_deriv.dims:
        raise ValueError(f"{event_dim!r} not found in norm_deriv.dims={norm_deriv.dims}")

    # Infer sample dimensions if not provided.
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

    # Rename to common internal sample dimension.
    S_std = S.rename({S_sample_dim: "sample"})
    y_std = norm_deriv.rename({y_sample_dim: "sample"})

    # Align on common coordinates.
    S_std, y_std = xr.align(S_std, y_std, join="inner")

    if event_index < 0 or event_index >= S_std.sizes[event_dim]:
        raise ValueError(
            f"event_index must be in [0, {S_std.sizes[event_dim] - 1}], got {event_index}"
        )

    n_samples = S_std.sizes["sample"]
    if p < 2 or p > n_samples:
        raise ValueError(f"p must be in [2, {n_samples}], got {p}")

    # Tau-hat is expected to be a vector over k.
    tau_hat_np = np.asarray(
        tau_hat.data if isinstance(tau_hat, xr.DataArray) else tau_hat,
        dtype=float,
    )
    if tau_hat_np.ndim != 1:
        raise ValueError("tau_hat must be 1D.")

    if k_end is None:
        k_end = tau_hat_np.size - 1
    if k_end < 0:
        raise ValueError(f"k_end must be >= 0, got {k_end}")
    if tau_hat_np.size != k_end + 1:
        raise ValueError(
            f"tau_hat length must equal k_end + 1. Got len(tau_hat)={tau_hat_np.size}, "
            f"k_end={k_end}."
        )
    if k_end > n_samples - p:
        raise ValueError(f"k_end must be in [0, {n_samples - p}], got {k_end}")

    # Parameters from Appendix A.
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

    # Time axis used in the fit.
    # This should match the time axis used in mle_tau_moteki_kondo.
    t = np.arange(n_samples, dtype=float) * h

    # Select the requested event only.
    s_event = np.asarray(S_std.sel({event_dim: event_index}).values, dtype=float)
    y_event = np.asarray(y_std.sel({event_dim: event_index}).values, dtype=float)

    if not (np.all(np.isfinite(s_event)) and np.all(np.isfinite(y_event))):
        raise ValueError(f"Selected event_index={event_index} contains non-finite values.")

    d2_vals = np.full(k_end + 1, np.nan, dtype=float)

    for k in range(k_end + 1):
        # Consecutive p-point subset starting at k, exactly as in the tau_hat search.
        yk = y_event[k : k + p]
        sk = s_event[k : k + p]
        tk = t[k : k + p]

        if not (np.all(np.isfinite(yk)) and np.all(np.isfinite(sk))):
            continue

        tau_k = float(tau_hat_np[k])
        if not np.isfinite(tau_k):
            continue

        # Build the subset mean/covariance using the same equations as the MLE.
        # Mean: Eq. (A.4)
        # Covariance: Eqs. (A.10a), (A.10b)
        # Statistical distance: Eq. (A.11)
        _, _, _, d2 = _moteki_kondo_subset_statistics(
            yk,
            sk,
            tk,
            tau_k,
            h=h,
            sigma_bar=sigma_bar,
            delta_sigma=delta_sigma,
            A1=A1,
            A2=A2,
            A3=A3,
        )

        if d2 is not None and np.isfinite(d2):
            d2_vals[k] = d2

    out = xr.DataArray(
        d2_vals,
        dims=("k",),
        coords={"k": np.arange(k_end + 1)},
        name="d2",
        attrs={
            "long_name": f"Moteki & Kondo statistical distance d^2(k) for event_index={event_index}",
            "units": "dimensionless",
        },
    )
    return out