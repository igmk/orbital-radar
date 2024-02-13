"""
Plot weighting functions.
"""

import matplotlib.pyplot as plt


def plot_range_weighting_function(sub):
    """
    Plot range weighting function.
    """

    fig, ax = plt.subplots(1, 1)

    ax.set_title("Range weighting function")

    ax.plot(
        sub.beam.range_weights,
        sub.beam.range_bins,
        marker=".",
        color="coral",
        label="Weighting function",
    )

    ax.axhline(-sub.beam.spec.pulse_length / 2, color="k", linestyle="--")
    ax.axhline(
        sub.beam.spec.pulse_length / 2,
        color="k",
        linestyle="--",
        label="Pulse length",
    )
    ax.axhline(y=0, color="k", label="Pulse center")

    ax.set_xlim(left=0)

    ax.legend(frameon=False)

    ax.set_xlabel("Response")
    ax.set_ylabel("Range wrt. beam width [m]")

    return fig


def plot_along_track_weighting_function(sub):
    """
    Plot along track weighting function
    """

    fig, ax = plt.subplots(1, 1)

    ax.set_title("Along-track weighting function")

    ax.plot(
        sub.beam.atrack_bins,
        sub.beam.atrack_weights,
        marker=".",
        color="coral",
        label="Weighting function",
    )

    ax.axvline(-sub.beam.ifov / 2, color="k", linestyle="--")
    ax.axvline(sub.beam.ifov / 2, color="k", linestyle="--", label="IFOV")
    ax.axvline(x=0, color="k", label="Line of sight")

    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, bbox_to_anchor=(0.1, 0.3))

    ax.set_ylabel("Response")
    ax.set_xlabel("Distance from line of sight [m]")

    return fig
