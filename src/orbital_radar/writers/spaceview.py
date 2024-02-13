"""
This function writes the output of the orbital radar simulator to a NetCDF 
file.
"""

import os

import xarray as xr


def write_spaceview(ds, filename):
    """
    Writes dataset to NetCDF file at the user-specified directory.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing the output of the orbital radar simulator
    filename: str
        Name of the output file
    """

    ds.to_netcdf(filename, mode="w")

    print(f"Written file: {filename}")
