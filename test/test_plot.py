"""
Test plot routines
"""

import orbital_radar as ora
from orbital_radar.plotting.noise import plot_noise
from orbital_radar.plotting.weighting import (
    plot_along_track_weighting_function,
    plot_range_weighting_function,
)

# simulation used for all plot tests
sub = ora.Suborbital(
    geometry="groundbased",
    name="min",
    config_file="orbital_radar_config.toml",
    suborbital_radar="inoe94",
    input_radar_format="geoms",
)

sub.run_date(date="2022-02-14", write_output=False)


class TestPlot:
    def test_along_track(self):
        """
        Run groundbased simulator on a specific test date and run along track
        plot function
        """

        fig = sub.plot(a0=0, a1=5 * 1e4)

        assert fig is not None

    def test_histogram(self):
        """
        Run groundbased simulator on a specific test date and run histogram
        plot function
        """

        fig = sub.plot_histogram(
            variables=["ze", "vm", "ze_sat", "vm_sat"],
            vmax=[1200, 1200, 70, 70],
            h=[0, 10e3],
        )

        assert fig is not None

    def test_scatter(self):
        """
        Test scatter plot
        """

        fig = sub.plot_scatter(var_ze="ze_sat_noise", var_vm="vm_sat_folded")

        assert fig is not None

    def test_scatter_aircraft(self):
        """
        Test scatter plot for aircraft
        """

        fig = sub.plot_scatter(var_ze="ze_sat_noise", var_vm="vm_sat_folded")

        assert fig is not None

    def test_plot_range_weighting_function(self):
        """
        Test range weighting function plot
        """

        fig = plot_range_weighting_function(sub)

        assert fig is not None

    def test_plot_along_track_weighting_function(self):
        """
        Test along-track weighting function plot
        """

        fig = plot_along_track_weighting_function(sub)

        assert fig is not None

    def test_plot_noise(self):
        """
        Test noise plot
        """

        fig = plot_noise(sub)
        assert fig is not None
