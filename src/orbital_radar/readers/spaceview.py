"""
This script reads the output of the simulator
"""

import xarray as xr


def read_spaceview(file):
    """
    Reads the output of the simulator.

    Parameters
    ----------
    file: str
        Path to the NetCDF file.

    Returns
    -------
    ds: xarray.Dataset
        Dataset containing the output of the simulator.
    """

    with xr.open_dataset(file) as ds:
        ds.load()

    return ds
