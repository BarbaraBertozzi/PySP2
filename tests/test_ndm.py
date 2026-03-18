import pysp2
import numpy as np
np.set_printoptions(threshold=np.inf)

from pysp2.util.normalized_derivative_method import MLEConfig, mle_tau_moteki_kondo

def test_central_difference():
    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False)
    dSdt = pysp2.util.central_difference(my_binary, normalize=False, baseline_to_zero=False)
    
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=5876, time=0).item(), 
                                   8.3333333333e6, decimal=2)
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=5876, time=99).item(), 
                                   7.166666666e7, decimal=2)
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=5876, time=19).item(), 
                                   1.5e7, decimal=2)
    assert np.isfinite(dSdt).all()

    dSdt_norm = pysp2.util.central_difference(my_binary, normalize=True, baseline_to_zero=False)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch4'].isel(event_index=5876, time=0).item(), 
                                   8.3333333333e6/-30168, decimal=2)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch4'].isel(event_index=5876, time=99).item(), 
                                   7.166666666e7/-30152, decimal=2)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch4'].isel(event_index=5876, time=19).item(), 
                                   1.5e7/-30132, decimal=2)
    
def test_mle_estimate_tau():
    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False, baseline_to_zero=True)
    dSdt = pysp2.util.central_difference(my_binary, normalize=False, baseline_to_zero=True)

    cfg = MLEConfig(
    h=0.4e-6,           # example: 0.4 microseconds
    sigma_bar=16.6,   # example; use your measured average width
    delta_sigma=1.2,# example; use your measured width std dev
    A1=0.37,
    A2=1.6e-2,
    A3=6.2e-4,
)
    print("my_binary", my_binary)
    print("my_binary time coordinates:", my_binary['time'].isel(event_index=499).values)
    print("my_binary['Data_ch0'].isel(event_index=499).values", my_binary['Data_ch0'].isel(event_index=499).values)
    print("time of max signal:", my_binary['Data_ch0'].isel(event_index=499).argmax().item())

    # One event
    tau_one = mle_tau_moteki_kondo(
        S=my_binary["Data_ch0"],
        norm_deriv=dSdt["Data_ch0"],
        p=20,
        event_index=499,
        config=cfg,
    )
    print(tau_one)

    # All events
    #tau_all = mle_tau_moteki_kondo(
    #    S=my_binary["Data_ch0"],
    #    norm_deriv=dSdt["Data_ch0"],
    #    p=11,
    #    config=cfg,
    #)
    #tau_hat = pysp2.util.mle_tau_moteki_kondo(my_binary['Data_ch0'].isel(event_index=5876), dSdt['Data_ch0'].isel(event_index=5876), 
    #                                          11, tau_grid=np.arange(-10, 10.1, 0.1), config=cfg)
    #print(tau_hat)
    
    #np.testing.assert_almost_equal(tau_hat.isel(k=0).item(), -0.5, decimal=1)
    #np.testing.assert_almost_equal(tau_hat.isel(k=10).item(), -0.5, decimal=1)
    #np.testing.assert_almost_equal(tau_hat.isel(k=20).item(), -0.5, decimal=1)
    #np.testing.assert_almost_equal(tau_hat.isel(k=30).item(), -0.5, decimal=1)
    