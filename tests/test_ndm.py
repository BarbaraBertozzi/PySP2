import pysp2
import numpy as np

def test_central_difference():
    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False)
    dSdt, norm_deriv = pysp2.util.central_difference(my_binary)
    
    # Check that the outputs have the expected dimensions and contain finite values
    assert dSdt.dims == my_binary.dims
    assert norm_deriv.dims == my_binary.dims
    assert np.isfinite(dSdt).all()
    assert np.isfinite(norm_deriv).all()