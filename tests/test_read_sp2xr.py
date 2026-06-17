import numpy as np

import pysp2


def test_read_sp2xr_hk_file():
    hk = pysp2.io.read_sp2xr_hk_file(pysp2.testing.EXAMPLE_SP2XR_HK)
    assert hk.sizes['time'] == 20650
    assert 'Laser TEC Temp' in hk.data_vars
    np.testing.assert_almost_equal(
        float(hk['Cavity Pressure'].mean()), 288.88, decimal=2)


def test_read_sp2xr():
    ds = pysp2.io.read_sp2xr(pysp2.testing.EXAMPLE_SP2XR_SP2B)
    assert ds.sizes['event_index'] == 457
    assert ds.sizes['columns'] == 4096
    assert 'Data_ch0' in ds.data_vars
    assert 'Data_ch1' in ds.data_vars
    assert ds['Data_ch0'].isel(event_index=0).max().item() > 6_000_000


def test_read_sp2xr_pbp():
    ds = pysp2.io.read_sp2xr_pbp(pysp2.testing.EXAMPLE_SP2XR_PBP)
    assert ds.sizes['event_index'] == 341715
    # The matching HK file lives alongside the PbP file, so an absolute
    # 'time' coordinate should be derived automatically.
    assert 'time' in ds.coords
    # Peak-fit outputs are exposed under PySP2-style names.
    assert 'PkHt_ch0' in ds.data_vars
    assert 'PkHt_ch1' in ds.data_vars
    # Firmware-calibrated columns are dropped by default.
    assert 'Scatter Size (nm)' not in ds.data_vars
    assert 'Incand Mass (fg)' not in ds.data_vars
