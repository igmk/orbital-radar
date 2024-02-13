"""
Reads weighting function from EarthCARE CPR.
"""

import numpy as np
import xarray as xr


def read_range_weighting_function(file):
    """
    Reads EarthCARE CPR range weighting function. The pulse length factor
    is reversed to match the sign convention of the groundbased radar.

    Parameters
    ----------
    file : str
        Path to file containing weighting function

    Returns
    -------
    wf : xarray.Dataset
        Weighting function
    """

    wf = np.loadtxt(file)

    ds_wf = xr.Dataset()
    ds_wf.coords["tau_factor"] = -wf[:, 0]
    ds_wf["response"] = ("tau_factor", wf[:, 1])

    ds_wf.tau_factor.attrs = dict(
        long_name="pulse length factor",
        short_name="tau_factor",
        description="multiply by tau to get height relative to pulse center",
    )

    ds_wf.response.attrs = dict(
        long_name="weighting function",
        short_name="response",
        units="dB",
        description="weighting function for CPR",
    )

    return ds_wf
