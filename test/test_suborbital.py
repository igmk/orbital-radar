"""
Test groundbased simulator
"""

import orbital_radar as ora


class TestSuborbital:
    def test_groundbased_94ghz(self):
        """
        Run groundbased simulator on a specific test date with 94 GHz radar
        """

        sub = ora.Suborbital(
            geometry="groundbased",
            name="min",
            config_file="orbital_radar_config.toml",
            suborbital_radar="inoe94",
            input_radar_format="geoms",
        )

        sub.run_date(date="2022-02-14")

        assert sub.ds is not None
        assert "ze" in sub.ds.data_vars

    def test_groundbased_35ghz(self):
        """
        Run groundbased simulator on a specific test with 35 GHz radar
        """

        sub = ora.Suborbital(
            geometry="groundbased",
            name="jue",
            config_file="orbital_radar_config.toml",
            suborbital_radar="mira",
            input_radar_format="cloudnet",
        )

        sub.run_date(date="2021-04-01")

        assert sub.ds is not None
        assert "ze" in sub.ds.data_vars

    def test_groundbased_arm(self):
        """
        Run groundbased simulator for an ARM radar. This test differs from
        other groundbased setups as ARM data is a function of height and not
        range.
        """

        sub = ora.Suborbital(
            geometry="groundbased",
            name="arm",
            config_file="orbital_radar_config.toml",
            suborbital_radar="arm",
            input_radar_format="arm",
        )

        sub.run_date(date="2009-11-13")

        assert sub.ds is not None
        assert "ze" in sub.ds.data_vars

    def test_airborne(self):
        """
        Run airborne simulator on a specific test date
        """

        sub = ora.Suborbital(
            geometry="airborne",
            name="mp5",
            config_file="orbital_radar_config.toml",
            suborbital_radar="mirac",
            input_radar_format="mirac_p5",
        )

        sub.run_date(date="2019-04-01")

        assert sub.ds is not None
        assert "ze" in sub.ds.data_vars

    def test_airborne_real_flight_velocity(self):
        """
        Test airborne simulator with real flight velocity instead of constant
        velocity.
        """

        sub = ora.Suborbital(
            geometry="airborne",
            name="rasta",
            config_file="orbital_radar_config.toml",
            suborbital_radar="rasta",
            input_radar_format="rasta",
        )

        sub.run_date(date="2022-08-06")

        assert sub.ds is not None
        assert "ze" in sub.ds.data_vars

    def test_pamtra(self):
        """
        Test transformation of PAMTRA-ICON simulation for a groundbased
        location.
        """

        sub = ora.Suborbital(
            geometry="groundbased",
            name="pamtra",
            config_file="orbital_radar_config.toml",
            suborbital_radar="joyrad94",
            input_radar_format="pamtra",
        )

        sub.run_date(date="2022-01-13")

        assert sub.ds is not None
        assert "ze" in sub.ds.data_vars
