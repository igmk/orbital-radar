"""
This script tests the simulator by simulating mock data.
"""

import numpy as np
import pandas as pd
import xarray as xr

import orbital_radar as ora
from orbital_radar.helpers import db2li, li2db


class TestSimulator:
    def test_simulator(self):
        """
        Run simulator on specific test data.

        Input signal: 950 to 1050 m height and 500 to 1000 m along track
        Ze 20 dBZ, Vm 2 m/s

        EarthCARE CPR pixel: 1000 m height and 750 m along track with a height
        resolution of 100 m and an along-track resolution of 500 m
        """

        ds = xr.Dataset()
        ds.coords["along_track"] = np.arange(0, 1500, 100)
        ds.coords["height"] = np.arange(0, 2000, 10)
        ds.coords["time"] = (
            "along_track",
            pd.date_range(
                "2000-01-01", periods=len(ds["along_track"]), freq="S"
            ),
        )
        ds["ze"] = xr.DataArray(
            np.zeros((len(ds.along_track), len(ds.height))),
            dims=["along_track", "height"],
        )
        ds["vm"] = xr.DataArray(
            np.zeros((len(ds.along_track), len(ds.height))),
            dims=["along_track", "height"],
        )

        # add a blob of reflectivity and doppler velocity
        ze_in = 20
        vm_in = 2
        along_track0 = 500
        along_track1 = 1000
        height0 = 950
        height1 = 1050

        along_track0_idx = ds.along_track.to_index().get_loc(along_track0)
        along_track1_idx = ds.along_track.to_index().get_loc(along_track1)
        height0_idx = ds.height.to_index().get_loc(height0)
        height1_idx = ds.height.to_index().get_loc(height1)

        ds["ze"][
            along_track0_idx : along_track1_idx + 1,
            height0_idx : height1_idx + 1,
        ] = db2li(ze_in)
        ds["vm"][
            along_track0_idx : along_track1_idx + 1,
            height0_idx : height1_idx + 1,
        ] = vm_in
        sim = ora.Simulator(sat_name="earthcare")

        sim.transform(ds)

        # check height-gradient of along-track convolution
        assert (
            li2db(
                sim.ds["ze_acon"].sel(
                    height=slice(height0, height1),
                    along_track=slice(along_track0, along_track1),
                )
            ).diff("height")
            == 0
        ).all()

        assert (
            sim.ds["vm_acon"]
            .sel(
                height=slice(height0, height1),
                along_track=slice(along_track0, along_track1),
            )
            .diff("height")
            == 0
        ).all()

        assert (
            sim.ds["vm_acon_err"]
            .sel(
                height=slice(height0, height1),
                along_track=slice(along_track0, along_track1),
            )
            .diff("height")
            == 0
        ).all()

        assert (
            li2db(
                sim.ds["ze_aconint"].sel(
                    height=slice(height0, height1),
                    along_track_sat=slice(along_track0, along_track1),
                )
            ).max()
            > 0.7 * ze_in
        )

        assert (
            sim.ds["vm_aconint"]
            .sel(
                height=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            .max()
            > 0.7 * vm_in
        )

        assert (
            sim.ds["vm_aconint_err"]
            .sel(
                height=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            .max()
            > 0.7 * vm_in
        )

        # check if the corresponding satellite pixel contains high ze
        assert (
            li2db(
                sim.ds["ze_sat"].sel(
                    height_sat=slice(height0, height1),
                    along_track_sat=slice(along_track0, along_track1),
                )
            )
            > 0.5 * ze_in
        )

        # check if the corresponding satellite pixel contains high vm
        assert (
            sim.ds["vm_sat"].sel(
                height_sat=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            > 0.2 * vm_in
        )

        assert (
            sim.ds["vm_sat_vel"].sel(
                height_sat=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            > 0.2 * vm_in
        )

        assert (
            li2db(
                sim.ds["ze_sat_noise"].sel(
                    height_sat=slice(height0, height1),
                    along_track_sat=slice(along_track0, along_track1),
                )
            )
            > 0.5 * ze_in
        )

        assert (
            sim.ds["vm_sat_noise"].sel(
                height_sat=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            > 0.01 * vm_in
        )

        assert (
            sim.ds["vm_sat_folded"].sel(
                height_sat=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            > 0.1 * vm_in
        )

        # check quality flags
        assert (
            sim.ds["folding_flag"].sel(
                height_sat=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            == 0
        )

        assert (
            sim.ds["ms_flag"].sel(
                height_sat=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            == 0
        )

        assert (
            sim.ds["signal_fraction"].sel(
                height_sat=slice(height0, height1),
                along_track_sat=slice(along_track0, along_track1),
            )
            == 1
        )

        # check other parameters
        assert sim.ds["sat_range_resolution"] == sim.beam.spec.range_resolution
        assert (
            sim.ds["sat_along_track_resolution"]
            == sim.beam.spec.along_track_resolution
        )
        assert sim.ds["sat_ifov"] == sim.beam.ifov
