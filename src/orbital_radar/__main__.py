"""
Command line interface to orbital-radar.

This module provides a command line interface to orbital-radar. It is
invoked when the user types `orbital-radar` in the command line. The
user can choose to simulate a groundbased or airborne radar.

Get help on the command line interface by typing `orbital-radar -h` or
`orbital-radar --help` in the command line.
"""

import argparse
import sys

import numpy as np

from orbital_radar import __version__
from orbital_radar.suborbital import Suborbital


def main():
    """
    Command line interface to orbital-radar.
    """

    parser = argparse.ArgumentParser(
        description="Sub-orbital-to-orbital transformation tool for radar "
        f"version {__version__}"
    )
    parser.add_argument(
        "-g",
        "--geometry",
        type=str,
        choices=["groundbased", "airborne"],
        help="observation geometry of radar (groundbased or airborne)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        choices=["bco", "jue", "nor", "mag", "min", "nya", "mp5"],
        help="name of groundbased site or airborne platform (abbreviated)",
    )
    parser.add_argument(
        "-d",
        "--date",
        type=np.datetime64,
        help="date in the format yyyy-mm-dd (e.g., 2023-09-22)",
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=np.datetime64,
        help="start date in the format yyyy-mm-dd (e.g., 2023-09-22)",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=np.datetime64,
        help="end date in the format yyyy-mm-dd (e.g., 2023-09-30)",
    )
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        help="path to the configuration file that contains the site-"
        "dependent parameters and directory paths",
    )
    parser.add_argument(
        "-r",
        "--suborbital_radar",
        type=str,
        choices=["mirac", "joyrad94", "smhi94", "inoe94", "bco94", "mira"],
        help="name of the suborbital radar (e.g. mirac)",
    )
    parser.add_argument(
        "-i",
        "--input_radar_format",
        type=str,
        choices=[
            "cloudnet",
            "uoc_v0",
            "uoc_v1",
            "uoc_v2",
            "geoms",
            "bco",
            "mirac_p5",
        ],
        help=(
            "format of input NetCDF radar data (uoc_v0, uoc_v1, uoc_v2, "
            "uoc_geoms, bco, mirac_p5). Default is cloudnet. Otherwise, these "
            "formats might be correct: "
            "uoc_v0 for nya, "
            "uoc_v1 for jue, "
            "uoc_v2 for nor and mag, "
            "geoms for min, "
            "bco for bco, "
            "mirac_p5 for mp5."
        ),
        default="cloudnet",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"orbital-radar version {__version__}",
    )
    args = parser.parse_args()

    # make sure that either date or start_date and end_date are given
    if args.date is None and (
        args.start_date is None or args.end_date is None
    ):
        raise ValueError(
            "Either date or start_date and end_date must be given"
        )

    sub = Suborbital(
        geometry=args.geometry,
        name=args.name,
        config_file=args.config_file,
        suborbital_radar=args.suborbital_radar,
        input_radar_format=args.input_radar_format,
    )

    if args.date is not None:
        sub.run_date(args.date)

    else:
        sub.run(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    sys.exit(main())
