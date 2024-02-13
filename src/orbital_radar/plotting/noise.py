"""
Plot noise lookup tables.
"""

import matplotlib.pyplot as plt


def plot_noise(sub):
    """
    Plot noise lookup tables.
    """

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), constrained_layout=True
    )

    # ze uncertainty
    ax1.set_title("Ze uncertainty")
    ax1.plot(
        sub.beam.spec.ze_bins,
        sub.beam.spec.ze_std,
        marker="o",
        color="coral",
        label="std($Z_e$)",
    )

    # background
    ax1.axhline(
        sub.beam.spec.ze_std_background,
        color="coral",
        label="std($Z_e$) background",
    )

    # noise floor
    ax1.axvline(
        sub.beam.spec.noise_ze,
        color="coral",
        linestyle="--",
        label=r"$Z_e$ noise floor",
    )

    # doppler velocity uncertainty
    ax2.set_title("Vm uncertainty")
    ax2.plot(
        sub.beam.spec.vm_bins_broad,
        sub.beam.spec.vm_std_broad,
        marker="o",
        color="coral",
        label="std($v_{BROAD}$)",
    )

    # background
    ax2.axhline(
        sub.beam.spec.vm_std_broad_background,
        color="coral",
        label=r"std($v_{BROAD}$) background",
    )

    # add detection limit to both axis
    ax1.axvline(
        x=sub.beam.spec.detection_limit,
        color="k",
        linestyle="--",
        label="$Z_e$ detection limit",
    )
    ax2.axvline(
        x=sub.beam.spec.detection_limit,
        color="k",
        linestyle="--",
        label="$Z_e$ detection limit",
    )

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    ax1.set_ylabel("Ze uncertainty [dBZ]")
    ax2.set_ylabel("Vm uncertainty [m/s]")
    ax2.set_xlabel("Ze [dBZ]")

    ax1.legend()
    ax2.legend()

    return fig
