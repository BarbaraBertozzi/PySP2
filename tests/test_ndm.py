import pysp2
import numpy as np

from pysp2.util.normalized_derivative_method import plot_normalized_derivative

def test_central_difference():
    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False)
    dSdt = pysp2.util.central_difference(my_binary, normalize=False)
    
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=5876, time=0).item(), 
                                   8.3333333333e6, decimal=2)
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=5876, time=99).item(), 
                                   7.166666666e7, decimal=2)
    np.testing.assert_almost_equal(dSdt['Data_ch4'].isel(event_index=5876, time=19).item(), 
                                   1.5e7, decimal=2)
    assert np.isfinite(dSdt).all()

    dSdt_norm = pysp2.util.central_difference(my_binary, normalize=True)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch4'].isel(event_index=5876, time=0).item(), 
                                   8.3333333333e6/-30168, decimal=2)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch4'].isel(event_index=5876, time=99).item(), 
                                   7.166666666e7/-30152, decimal=2)
    np.testing.assert_almost_equal(dSdt_norm['Data_ch4'].isel(event_index=5876, time=19).item(), 
                                   1.5e7/-30132, decimal=2)
    