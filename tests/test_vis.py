import matplotlib
import pytest
import xarray as xr
import numpy as np

import pysp2
from pysp2.util.normalized_derivative_method import plot_normalized_derivative

matplotlib.use("Agg")

@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_normalized_derivative():

    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False)
    dSdt_norm = pysp2.util.central_difference(my_binary, normalize=True, baseline_to_zero=True)

    # Test the plotting function for channel 0 and record number 2
    ax = plot_normalized_derivative(dSdt_norm, record_no=499, chn=0)
    fig = ax.figure
    
    return fig