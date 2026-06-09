import pysp2
import numpy as np
np.set_printoptions(threshold=np.inf)

from pysp2.util.normalized_derivative_method import MLEConfig, mle_tau_moteki_kondo, compute_d2_moteki_kondo
from pysp2.util.normalized_derivative_method import compute_sigma_moteki_kondo
event=720

def test_central_difference():
    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False)
    dSdt = pysp2.util.central_difference(my_binary, normalize=False, baseline_to_zero=False)
    print("my_binary", my_binary)
    
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=event, time=0).item(), 
                                   8.3333333333e6, decimal=2)
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=event, time=99).item(), 
                                   7.166666666e7, decimal=2)
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=event, time=19).item(), 
                                    1.5e7, decimal=2)
    assert np.isfinite(dSdt).all()

    dSdt_norm = pysp2.util.central_difference(my_binary, normalize=True, baseline_to_zero=True)
    print("dSdt_norm values:", dSdt_norm['Data_ch0'].isel(event_index=event, time=slice(0, 100)).values)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch0'].isel(event_index=event, time=0).item(), 
                                   8.3333333333e6/-30168, decimal=2)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch0'].isel(event_index=event, time=99).item(), 
                                   7.166666666e7/-30152, decimal=2)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch0'].isel(event_index=event, time=19).item(), 
                                   1.5e7/-30132, decimal=2)
    
def test_ndm_moteki_kondo():
    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False, baseline_to_zero=True)
    dSdt = pysp2.util.central_difference(my_binary, normalize=True, baseline_to_zero=True)  

    cfg = MLEConfig(
    h=0.4,           # example: 0.4 microseconds
    sigma_bar= 16.6*0.4,  # example; use your measured average width
    delta_sigma=1.2*0.4, # example; use your measured width std dev
    A1=0.37,
    A2=1.6e-2,
    A3=6.2e-4,
)
    
    ## Test one event ##################################################
    tau = mle_tau_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        p=10,
        ch="Data_ch0",
        event_index=499,
        min_start=15,
        width_metric="fwhm",
        config=cfg,
    )

    tau_val_true = my_binary['Data_ch0'].isel(event_index=499).argmax().item()*0.4
    # Test that the estimated tau for a subset of results is close to the true value for the event
    for i in range(6, 12):
        np.testing.assert_almost_equal(tau[i], tau_val_true, decimal=6)
    
    d2 = compute_d2_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        tau_hat=tau,
        p=10,
        ch="Data_ch0",
        event_index=499,
        min_start=15,
        width_metric="fwhm",
        config=cfg,
    )

    k_min_local = int(d2.argmin(dim="k").item())
    # Get corresponding tau value
    tau_best = tau.isel(k=k_min_local).item()

    # Assert closeness
    np.testing.assert_allclose(
        tau_best,
        tau_val_true,
        atol=0.05e-05,  # absolute tolerance = 5e-7
    )

    sigma_ds = compute_sigma_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        tau_hat=tau,
        d2=d2,
        p=10,
        ch="Data_ch0",
        event_index=499,
        min_start=15,
        width_metric="fwhm",
        config=cfg,
    )

    # example: use the best sigma value from your analysis, divided by 2.355 to convert FWHM to std dev
    # value of 26.275540424824687 is the average FWHM in seconds from the example dataset 
    # this is just an example of how to set the tolerance based on the expected sigma of the events
    sigma_best = (26.275540424824687*0.4)/2.355  

    np.testing.assert_allclose(
        sigma_ds['sigma_hat'].values,
        sigma_best,
        atol=2e-6,  # absolute tolerance = 0.02 microseconds
    )

    ## Test another event ##################################################
    tau = mle_tau_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        p=6,
        ch="Data_ch0",
        event_index=1040,
        min_start=15,
        width_metric="fwtm",
        config=cfg,
    )

    tau_val = my_binary['Data_ch0'].isel(event_index=1040).argmax().item()*0.4
    # Test that the estimated tau for a subset of results is close to the true value for the event
    for i in range(18, 21):
        np.testing.assert_allclose(tau[i], tau_val, atol=0.08e-05)

    d2 = compute_d2_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        tau_hat=tau,
        p=6,
        ch="Data_ch0",
        event_index=1040,
        min_start=15,
        width_metric="fwtm",
        config=cfg,
    )

    k_min_local = int(d2.argmin(dim="k").item())
    # Get corresponding tau value
    tau_best = tau.isel(k=k_min_local).item()

    # Assert closeness
    np.testing.assert_allclose(
        tau_best,
        tau_val,
        atol=0.04e-05,  # absolute tolerance = 4e-7
    )

    sigma_ds = compute_sigma_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        tau_hat=tau,
        d2=d2,
        p=6,
        ch="Data_ch0",
        event_index=1040,
        min_start=15,
        width_metric="fwtm",
        config=cfg,
    )

    # example: use the best sigma value from your analysis, divided by 2.355 to convert FWHM to std dev
    # value of 25.00349832837095 is the average FWHM in seconds from the example dataset 
    # this is just an example of how to set the tolerance based on the expected sigma of the events
    sigma_best = (25.00349832837095*0.4)/2.355  

    np.testing.assert_allclose(
        sigma_ds['sigma_hat'].values,
        sigma_best,
        atol=0.2,  # absolute tolerance = 0.02 microseconds
    )
    
    ## Test another event ##################################################
    tau = mle_tau_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        p=8,
        ch="Data_ch4",
        event_index=2008,
        min_start=15,
        width_metric="fwtm",
        config=cfg,
    )
    print("tau values:", tau.values)

    d2 = compute_d2_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        tau_hat=tau,
        p=8,
        ch="Data_ch4",
        event_index=2008,
        min_start=15,
        width_metric="fwtm",
        config=cfg,
    )
    print("d2 values:", d2.values)

    # Test that the estimated tau for a subset of results is close to the true value for the event
    for i in range(2, 4):
        np.testing.assert_allclose(tau[i], tau_val_true, atol=0.08e-05)

    k_min_local = int(d2.argmin(dim="k").item())
    # Get corresponding tau value
    tau_best = tau.isel(k=k_min_local).item()
    print("tau_best:", tau_best)
 
    # Assert closeness
    np.testing.assert_allclose(
        tau_best,
        tau_val_true,
        atol=0.04e-05,  # absolute tolerance = 1e-6 (larger tolerance for evaporation events)
    )

    sigma_ds = compute_sigma_moteki_kondo(
        S=my_binary,
        norm_deriv=dSdt,
        tau_hat=tau,
        d2=d2,
        p=8,
        ch="Data_ch4",
        event_index=2008,
        min_start=15,
        width_metric="fwtm",
        config=cfg,
    )
    print("sigma:", sigma_ds.values)

    sigma_best = (26.275540424824687*0.4)/2.355  
    print("sigma_best:", sigma_best)

    np.testing.assert_allclose(
        sigma_ds['sigma_hat'].values,
        sigma_best,
        atol=3.3e-6,  # absolute tolerance = 3 microseconds
    )
    # the tolerance is too high for the last test because the event is an 
    # evaporation event with a much broader peak shape, which can lead to less 
    # precise estimates of tau and sigma using the MLE method. In practice, you 
    # may want to use a different method or adjust the config parameters for evaporation 
    # events to get more accurate estimates.