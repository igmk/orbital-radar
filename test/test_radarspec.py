"""
This script tests the radarspec module.
"""

import numpy as np

from orbital_radar.radarspec import RadarBeam
from orbital_radar.readers.config import read_config


class TestRadarSpec:
    def test_wavelength(self):
        """
        Test wavelength calculation
        """

        beam = RadarBeam(sat_name="earthcare")

        beam.spec.frequency = 10e9

        beam.calculate_wavelength()

        assert np.round(beam.wavelength, 2) == 0.03

    def test_earthcare_range_weighting_function(self):
        """
        Test range weighting function calculation of EarthCARE.
        """

        # get path to cpr range weighting function
        file = read_config("orbital_radar_config.toml")["spaceborne_radar"][
            "file_earthcare"
        ]

        beam = RadarBeam(sat_name="earthcare", file_earthcare=file)

        beam.calculate_weighting_functions(
            range_coords=np.arange(-500, 10000, 10),
            along_track_coords=np.arange(0, 6e5, 25),
        )

        assert beam.range_bins[0] == -beam.spec.pulse_length
        assert beam.range_weights[0] == 0
        assert (beam.range_weights.sum() - 1) < 1e-10

    def test_nyquist_velocity_from_prf(self):
        """
        Test Nyquist velocity calculation from pulse repetition frequency.
        """

        beam = RadarBeam(
            sat_name="earthcare",
            nyquist_from_prf=False,
            pulse_repetition_frequency=7150,
        )

        beam.calculate_nyquist_velocity()

        assert np.abs(beam.spec.nyquist_velocity - 5.7) < 0.1
