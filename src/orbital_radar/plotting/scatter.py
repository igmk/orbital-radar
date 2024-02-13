"""
Scatter plot between satellite data and suborbital data.

Suborbital data is first averaged onto the satellite grid similar to the
non-uniform beam filling flag.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

from orbital_radar.helpers import li2db


def plot_scatter(ds, var_ze, var_vm):
    """
    Scatter plot between satellite data and suborbital data.

    Left: Radar reflectivity
    Right: Doppler velocity

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the processed suborbital and CPR data
    var_ze : str
        Variable name for radar reflectivity to be plotted
    var_vm : str
        Variable name for Doppler velocity to be plotted
    """

    # resample suborbital data
    da_ze_res = resample(ds, variable="ze")
    da_vm_res = resample(ds, variable="vm")

    # create plot
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 3), constrained_layout=True
    )

    # point density
    # radar reflectivity
    x = li2db(da_ze_res.values.flatten())
    y = li2db(ds[var_ze].values.flatten())
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))

    # this makes sure that not all values are either zero or nan
    if np.sum(~is_nan) > 0 and not (x[~is_nan] == 0).all():
        x = x[~is_nan]
        y = y[~is_nan]

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        ix = np.argsort(z)

        im = ax1.scatter(
            x[ix],
            y[ix],
            c=z[ix],
            cmap="magma",
            lw=0,
            s=1,
        )
        fig.colorbar(im, ax=ax1, label="Density")

    # Doppler velocity
    x = da_vm_res.values.flatten()
    y = ds[var_vm].values.flatten()
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))

    # this makes sure that not all values are either zero or nan
    if np.sum(~is_nan) > 0 and not (x[~is_nan] == 0).all():
        x = x[~is_nan]
        y = y[~is_nan]

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        ix = np.argsort(z)

        im = ax2.scatter(
            x[ix],
            y[ix],
            c=z[ix],
            cmap="magma",
            lw=0,
            s=1,
        )
        fig.colorbar(im, ax=ax2, label="Density")

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.grid(True)

        # plot diagonal line
        ax.plot(
            ax.get_xlim(),
            ax.get_xlim(),
            color="black",
            linewidth=1,
        )

    ax1.set_xlabel("Ze [dBZ]")
    ax1.set_ylabel("CPR Ze [dBZ]")
    ax1.set_xlim([-40, 25])
    ax1.set_ylim([-40, 25])

    ax2.set_xlabel("vm [m/s]")
    ax2.set_ylabel("CPR vm [m/s]")
    ax2.set_xlim([-3.5, 3.5])
    ax2.set_ylim([-3.5, 3.5])

    return fig


def resample(ds, variable):
    """
    Resampling from high-resolution grid to low-resolution grid
    """

    # create labels for each satellite pixel (height_sat x along_track_sat)
    labels = np.arange(ds.height_sat.size * ds.along_track_sat.size).reshape(
        ds.ze_sat.shape
    )

    # calculate bin edges of satellite grid
    range_resolution = ds["height_sat"][1] - ds["height_sat"][0]
    along_track_resolution = (
        ds["along_track_sat"][1] - ds["along_track_sat"][0]
    )

    along_track_sat_bin_edges = np.append(
        ds["along_track_sat"] - along_track_resolution / 2,
        ds["along_track_sat"][-1] + along_track_resolution / 2,
    )

    range_sat_bin_edges = np.append(
        ds["height_sat"] - range_resolution / 2,
        ds["height_sat"][-1] + range_resolution / 2,
    )
    
    # assign satellite pixel label to each input pixel of suborbital radar
    ix_range = np.searchsorted(
        ds.height_sat.values,
        ds.height.values,
        side="left",
    )
    ix_along_track = np.searchsorted(
        ds.along_track_sat.values,
        ds.along_track.values,
        side="left",
    )

    # adjust index at first position
    ix_range[ix_range == 0] = 1
    ix_along_track[ix_along_track == 0] = 1

    ix_range = ix_range - 1
    ix_along_track = ix_along_track - 1

    ix_range, ix_along_track = np.meshgrid(
        ix_along_track, ix_range, indexing="ij"
    )
    labels_input_grid = labels[ix_range, ix_along_track]

    # calculate standard deviation of ze on input grid in dBZ
    # this is done with pandas for faster performance
    df = ds[variable].stack({"x": ("along_track", "height")}).to_dataframe()
    df["labels"] = labels_input_grid.flatten()
    df_mean = df[variable].groupby(df["labels"]).mean()

    # convert to xarray
    da_res = xr.DataArray(
        df_mean.values.reshape(labels.shape),
        dims=["along_track_sat", "height_sat"],
        coords={
            "along_track_sat": ds.along_track_sat,
            "height_sat": ds.height_sat,
        },
    )

    return da_res
