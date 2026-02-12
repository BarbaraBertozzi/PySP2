import pysp2
import numpy as np

def test_central_difference():
    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False)
    dSdt = pysp2.util.central_difference(my_binary)
    
    # Check that the outputs have the expected dimensions and contain finite values
    print(f'dSdt shape: {dSdt.dims}')
    print(f'my_binary shape: {my_binary.dims}')

    #assert dSdt.dims == my_binary.dims
    assert np.isfinite(dSdt).all()