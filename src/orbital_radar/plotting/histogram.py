"""
Plots histogram of radar reflectivity and Doppler velocity.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from orbital_radar.helpers import li2db


def plot_histogram(ds, variables, vmax, h, show=False):
    """
    Plots histogram in a 2x2 manner with variables as specified by the user.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing radar data.
    variable : str
        Name of the variable to plot. E.g. vm_sat
    vmax : list
        Maximum count of colorbar for each variable
    h : list
        Height limit of the plot
    show : bool
        If True, show the plot.
    """

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)

    for i, variable in enumerate(variables):
        ax = axes.flatten()[i]

        # get type of variable
        if "ze" == variable[:2]:
            variable_type = "ze"
        elif "vm" == variable[:2]:
            variable_type = "vm"
        else:
            raise ValueError("Variable type not recognized.")

        plot_single_histogram(ds, variable, variable_type, ax=ax, vmax=vmax[i], h=h)

    if show:
        plt.show()

    return fig


def plot_single_histogram(ds, variable, variable_type, ax, vmax, h):
    """
    Plots histogram of radar reflectivity or Doppler velocity.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing radar data.
    variable : str
        Name of the variable to plot. E.g. vm_sat
    variable_type : str
        Type of the variable. E.g. vm
    ax : matplotlib.axes
        Axes to plot on.
    vmax : int
        Maximum count of colorbar.
    h : list
        Height limit of the plot
    """

    da_hist = calculate_histogram(ds, variable, variable_type)

    im = ax.pcolormesh(
        da_hist[variable_type],
        da_hist.height * 1e-3,
        da_hist.T,
        cmap="jet",
        vmin=0,
        vmax=vmax,
    )

    ax.get_figure().colorbar(im, ax=ax, label="Counts")

    if variable_type == "ze":
        if "ze" != variable:
            ax.set_xlabel("CPR Ze [dBZ]")
        else:
            ax.set_xlabel("Ze [dBZ]")
        ax.set_xlim([-40, 25])
    elif variable_type == "vm":
        if "vm" != variable:
            ax.set_xlabel("CPR Vm [m s$^{-1}$]")
        else:
            ax.set_xlabel("Vm [m s$^{-1}$]")
        ax.set_xlim([-5.7, 5.7])
    
    ax.set_ylabel("Height [km]")
    ax.set_ylim(h)

def calculate_histogram(ds, variable, variable_type):
    """
    Calculate histogram
    """

    # ze or vm bins
    if variable_type == "ze":
        bin_edges = np.arange(-40, 55.1, 0.5)
    elif variable_type == "vm":
        bin_edges = np.arange(-5.7, 5.7, 0.1)
    else:
        raise ValueError("Variable type not recognized.")

    # get height variable
    var_height = ds[variable].dims[1]

    height_bins = ds[var_height].values

    # define edges between height bins
    range_step = height_bins[1] - height_bins[0]  # assumes constant height
    range_bin_edges = 0.5 * (height_bins[1:] + height_bins[:-1])
    range_bin_edges = np.append(
        range_bin_edges, np.array([range_bin_edges[-1] + range_step])
    )
    range_bin_edges = np.append(
        np.array([range_bin_edges[0] - range_step]), range_bin_edges
    )

    da = li2db(ds[variable].stack(x=ds[variable].dims))
    hist, _, _ = np.histogram2d(
        x=da,
        y=da[var_height],
        bins=[bin_edges, range_bin_edges],
        density=False,
    )
    hist[hist == 0] = np.nan

    # create xarray data array
    da_hist = xr.DataArray(
        hist,
        dims=(variable_type, "height"),
        coords={
            variable_type: (bin_edges[1:] + bin_edges[:-1]) / 2,
            "height": height_bins,
        },
    )

    return da_hist
