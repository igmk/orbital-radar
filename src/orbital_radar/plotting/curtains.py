"""
Along-track plots of radar.
"""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from orbital_radar.helpers import li2db

# visualization settings
FACECOLOR = "silver"  # background color of each axis (e.g. gray)

mpl.rcParams["axes.facecolor"] = FACECOLOR


def axis_label(ax, label):
    """
    This function annotates an axis label to describe the dataset
    """

    ax.annotate(
        label, xy=(0.5, 1), xycoords="axes fraction", ha="center", va="bottom"
    )


def colormap_vm():
    """
    Creates colorbar for Doppler velocity.

    The colorbar contains three colormaps:
    - autumn: -2 to 0
    - winter: 0 to 2
    - cool: 2 to 6

    Returns
    -------
    cmap_vm: matplotlib.colors.LinearSegmentedColormap
        Colormap for Doppler velocity
    """

    n_colors = 256

    colors = np.vstack(
        (
            plt.cm.autumn(np.linspace(0, 1, n_colors // 4)),
            plt.cm.winter(np.linspace(0, 1, n_colors // 4)),
            plt.cm.cool(np.linspace(0, 1, n_colors // 2)),
        )
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("vm_colormap", colors)

    # make discrete colormap
    bounds = np.arange(-2, 6 + 0.25, 0.25)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm

def colormap_vm_sat():
    """
    Creates colorbar for Doppler velocity.

    The colorbar contains three colormaps:
    - plasma: -6 to 0 
    - winter: 0 to 2
    - cool: 2 to 6

    Returns
    -------
    cmap_vm: matplotlib.colors.LinearSegmentedColormap
        Colormap for Doppler velocity
    """

    n_colors = 256

    # Define the color segments for each colormap
    plasma_colors = plt.cm.plasma(np.linspace(0, 1, n_colors//2))
    winter_colors = plt.cm.winter(np.linspace(0, 1, n_colors//4))
    cool_colors = plt.cm.cool(np.linspace(0, 1, n_colors//4))

    # Stack the colors to create the full colormap
    colors = np.vstack((plasma_colors, winter_colors, cool_colors))  

    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("vm_colormap_sat", colors)

    # make discrete colormap
    bounds = np.arange(-6, 6+0.5, 0.5) # Adjusted to cover the range from -6 to 6
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def colormap_ze():
    """
    Creates colorbar for radar reflectivity.
    """

    cmap = plt.cm.jet.copy()

    cmap.set_under("gray", 1.0)
    cmap.set_over("black", 1.0)

    # make discrete colormap
    bounds = np.arange(-35, 20, 2.5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def plot_along_track(
    ds,
    variables=None,
    figsize=(8, 6),
    labels=None,
    a0=None,
    a1=None,
    h0=None,
    h1=None,
    show=False,
):
    """
    Plot along-track radar data.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing the along-track data
    variables: list
        List of variables to plot
    figsize: tuple
        Size of the figure (width, height)
    labels: dict
        Dictionary with labels for each variable. By default, the long_name
        attribute of the variable is used.
    a0: float
        Start distance along track [m]
    a1: float
        End distance along track [m]
    h0: float
        Start height [m]
    h1: float
        End height [m]
    """

    if variables is None:
        variables = ["ze", "vm", "ze_sat", "vm_sat"]

    else:
        for var in variables:
            if var not in ds:
                raise ValueError(f"Variable {var} not found in dataset")

    if labels is None:
        labels = {x: ds[x].attrs["long_name"] for x in variables}

    else:
        for var in variables:
            if var not in labels:
                labels[var] = ds[var].attrs["long_name"]

    n_rows = len(variables)

    # get colormaps
    cmap_vm, norm_vm         = colormap_vm_sat()
    # cmap_vm_sat, norm_vm_sat = colormap_vm_sat()
    cmap_ze, norm_ze         = colormap_ze()

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )

    for i, var in enumerate(variables):
        axis_label(axes[i], labels[var])

        # this is (along_track, height) or (along_track_sat, height_sat)
        dim_x = ds[var].dims[0]
        dim_y = ds[var].dims[1]

        kwargs = {dim_x: slice(a0, a1), dim_y: slice(h0, h1)}
        da_x = ds[var][dim_x].sel({dim_x: slice(a0, a1)}) * 1e-3
        da_y = ds[var][dim_y].sel({dim_y: slice(h0, h1)}) * 1e-3

        if "ze" in var:
            im = axes[i].pcolormesh(
                da_x,
                da_y,
                li2db(ds[var].T.sel(**kwargs)),
                cmap=cmap_ze,
                norm=norm_ze,
                shading="nearest",
            )
            fig.colorbar(im, ax=axes[i], extend="both", label="Ze [dBZ]")

        elif "vm" in var:
            im = axes[i].pcolormesh(
                da_x,
                da_y,
                ds[var].T.sel(**kwargs),
                cmap=cmap_vm,
                norm=norm_vm,
                shading="nearest",
            )
            fig.colorbar(im, ax=axes[i], extend="both", label="vm [m s$^{-1}$]")

        elif "flag" in var:
            im = axes[i].pcolormesh(
                da_x,
                da_y,
                ds[var].T.sel(**kwargs),
                cmap="viridis",
                vmin=0,
                vmax=1,
                shading="nearest",
            )
            fig.colorbar(im, ax=axes[i], label="Flag value")

        elif "signal_fraction" in var:
            im = axes[i].pcolormesh(
                da_x,
                da_y,
                ds[var].T.sel(**kwargs),
                cmap="viridis",
                vmin=0,
                vmax=1,
                shading="nearest",
            )
            fig.colorbar(im, ax=axes[i], label="Signal fraction")

        elif "nubf" in var:
            im = axes[i].pcolormesh(
                da_x,
                da_y,
                ds[var].T.sel(**kwargs),
                cmap="viridis",
                vmax=3,
                shading="nearest",
            )
            fig.colorbar(im, ax=axes[i], label="NUBF [dBZ]")

        else:  # any other variable is plotted like this
            im = axes[i].pcolormesh(
                da_x,
                da_y,
                ds[var].T.sel(**kwargs),
                cmap="viridis",
                shading="nearest",
            )
            fig.colorbar(im, ax=axes[i], label="Value")

    for ax in axes:
        ax.set_ylabel("Height [km]")
    axes[-1].set_xlabel("Distance along track [km]")

    for ax in axes[:-1]:
        ax.xaxis.set_tick_params(which="both", labelbottom=True)

    if show:
        plt.show()

    return fig
