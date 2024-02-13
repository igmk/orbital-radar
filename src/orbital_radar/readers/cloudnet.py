"""
This script contains functions to read cloudnet data.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

FILENAMES = {
    "cloudnet_ecmwf": "ecmwf",
    "cloudnet_categorize": "categorize",
}


def read_cloudnet(
    attenuation_correction_input, date, site_name, path, add_date=True
):
    """
    Reads Cloudnet data.

    The following file naming is expected (e.g. for 2022-02-14 at Mindelo):
    20220214_mindelo_ecmwf.nc
    20220214_mindelo_categorize.nc

    Parameters
    ----------
    attenuation_correction_input: str
        Cloudnet product to read. Either 'categorize' or 'ecmwf'.
    date: np.datetime64
        Date for which data is read.
    site_name: str
        Name of the site.
    path: str
        Path to the Cloudnet data. The path should contain the year, month, and
        day as subdirectories.
    add_date: bool, optional
        If True, the date is added to the path. Default is True.

    Returns
    -------
    ds: xarray.Dataset
        Cloudnet data.
    """

    if add_date:
        path = os.path.join(
            path,
            pd.Timestamp(date).strftime(r"%Y"),
            pd.Timestamp(date).strftime(r"%m"),
            pd.Timestamp(date).strftime(r"%d"),
        )

    if not os.path.exists(path):
        print(f"Warning: The cloudnet data path {path} does not exist")
        print("Warning: No attenuation correction will be applied")

        return None

    files = glob(
        os.path.join(path, f"*{FILENAMES[attenuation_correction_input]}.nc")
    )

    # return none if no files are found
    if len(files) == 0:
        # print warning
        print(
            f"No {attenuation_correction_input} Cloudnet files found "
            f"for {date} at "
            f"{site_name}"
        )

        return None

    # warn if more than one file is found
    if len(files) > 1:
        print(
            f"More than one {attenuation_correction_input} Cloudnet file "
            f"found for "
            f"{date} at {site_name}. Reading first file."
        )

    file = files[0]

    print(f"Reading {attenuation_correction_input} Cloudnet data: {file}")

    # model_time unit for older cloudnetpy versions in bad format
    if attenuation_correction_input == "cloudnet_categorize":
        ds = xr.open_dataset(file, decode_times=False)

        if (
            ds["model_time"].units == "decimal hours since midnight"
            or ds["model_time"].units == f"hours since {str(date)} +00:00"
        ):
            # model time
            ds = convert_time(
                ds=ds,
                time_variable="model_time",
                base_time=np.datetime64(date),
                factor=60 * 60 * 1e9,
            )

            # radar time
            ds = convert_time(
                ds=ds,
                time_variable="time",
                base_time=np.datetime64(date),
                factor=60 * 60 * 1e9,
            )

        # make sure that difference between first and last time is more than 12 h
        if (
            ds["model_time"].values[-1] - ds["model_time"].values[0]
        ) < np.timedelta64(12, "h"):
            print(
                f"Warning: The time difference between the first and last time "
                f"step is less than 12 hours for {date} at {site_name}. "
                f"Check if time format is being read correctly."
            )

            return None

        if (ds["time"].values[-1] - ds["time"].values[0]) < np.timedelta64(
            12, "h"
        ):
            print(
                f"Warning: The time difference between the first and last time "
                f"step is less than 12 hours for {date} at {site_name}. "
                f"Check if time format is being read correctly."
            )

            return None

    # problem did not occur for ecmwf data
    else:
        
        ds = xr.open_dataset(file)

    return ds


def convert_time(ds, time_variable, base_time, factor=1):
    """
    Convert time in seconds since base_time to datetime64.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the time variable.
    time_variable : str
        Name of the time variable.
    base_time : str
        Base time as string (e.g. "1970-01-01")
    factor : float, optional
        Factor to convert time to nanoseconds. Default is 1.
    """

    ds[time_variable] = (ds[time_variable] * factor).astype(
        "timedelta64[ns]"
    ) + np.datetime64(base_time)

    return ds
