r"""
Test radar reader functions for different formats. This test requires local
example data for each reader function for the tested site and date.

The test requires that all paths are specified in orbital_radar_config.toml
with the environment variable ``ORBITAL_RADAR_CONFIG_PATH`` pointing to the 
config file.
"""

import pandas as pd

from orbital_radar.readers.cloudnet import read_cloudnet
from orbital_radar.readers.config import read_config
from orbital_radar.readers.radar import Radar
from orbital_radar.readers.rangewf import read_range_weighting_function


class TestCloudnet:
    def test_cloudnet_categorize(self):
        """
        Test Cloudnet categorize reader
        """

        # get path to cloudnet data
        path = read_config("orbital_radar_config.toml")["paths"]["min"][
            "cloudnet"
        ]

        ds = read_cloudnet(
            attenuation_correction_input="cloudnet_categorize",
            date=pd.Timestamp("2022-02-14"),
            site_name="min",
            path=path,
        )

        assert "radar_gas_atten" in list(ds.keys())

    def test_cloudnet_ecmwf(self):
        """
        Test Cloudnet ECMWF reader
        """

        # get path to cloudnet data
        path = read_config("orbital_radar_config.toml")["paths"]["min"][
            "cloudnet"
        ]

        ds = read_cloudnet(
            attenuation_correction_input="cloudnet_ecmwf",
            date=pd.Timestamp("2022-02-14"),
            site_name="min",
            path=path,
        )

        assert "gas_atten" in list(ds.keys())


class TestRadar:
    def test_radar_uoc_v0(self):
        """
        Test UoC reader for Ny-Alesund file format
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["nya"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2021-02-07"),
            site_name="nya",
            path=path,
            input_radar_format="uoc_v0",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_radar_uoc_v1(self):
        """
        Test UoC reader.
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["jue"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2021-03-18"),
            site_name="jue",
            path=path,
            input_radar_format="uoc_v1",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_radar_geoms(self):
        """
        Test UoC GEOMS reader
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["min"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2022-02-14"),
            site_name="min",
            path=path,
            input_radar_format="geoms",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_bco(self):
        """
        Test BCO reader
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["bco"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2019-11-01"),
            site_name="bco",
            path=path,
            input_radar_format="bco",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_uoc_v2(self):
        """
        Test UoC reader version 2
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["nor"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2022-10-10"),
            site_name="nor",
            path=path,
            input_radar_format="uoc_v2",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_arm(self):
        """
        Test ARM reader.
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["arm"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2009-11-13"),
            site_name="arm",
            path=path,
            input_radar_format="arm",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_cloudnet(self):
        """
        Test cloudnet radar reader
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["jue"][
            "cloudnet"
        ]

        rad = Radar(
            date=pd.Timestamp("2021-04-01"),
            site_name="jue",
            path=path,
            input_radar_format="cloudnet",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_polar5(self):
        """
        Test UoC reader version 2
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["mp5"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2019-04-01"),
            site_name="mp5",
            path=path,
            input_radar_format="mirac_p5",
        )

        assert "ze" in list(rad.ds_rad.keys())

    def test_rasta(self):
        """
        Test RASTA reader
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["rasta"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2022-08-06"),
            site_name="rasta",
            path=path,
            input_radar_format="rasta",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())

    def test_pamtra(self):
        """
        Test PAMTRA reader
        """

        # get path to radar data
        path = read_config("orbital_radar_config.toml")["paths"]["pamtra"][
            "radar"
        ]

        rad = Radar(
            date=pd.Timestamp("2022-08-01"),
            site_name="pamtra",
            path=path,
            input_radar_format="pamtra",
        )

        assert "ze" in list(rad.ds_rad.keys())
        assert "vm" in list(rad.ds_rad.keys())


class TestConfig:
    def test_config_reader(self):
        """
        Test reading of config file
        """

        config = read_config("orbital_radar_config.toml")

        assert type(config) == dict


class TestRangeWeightingFunction:
    def test_earthcare(self):
        """
        Test reading of EarthCARE range weighting function
        """

        # get file from config
        file = read_config("orbital_radar_config.toml")["spaceborne_radar"][
            "file_earthcare"
        ]

        ds_wf = read_range_weighting_function(file)

        assert "response" in list(ds_wf)
