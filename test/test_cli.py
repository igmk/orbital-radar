"""
Test command line interface (orbital-radar)
"""

import subprocess


class TestCLI:
    def test_cli_single_day(self):
        """
        Test orbital-radar for a single day
        """

        s = subprocess.run(
            [
                "orbital-radar",
                "-g",
                "groundbased",
                "-n",
                "jue",
                "-d",
                "2021-04-01",
                "-f",
                "orbital_radar_config.toml",
                "-r",
                "mira",
                "-i",
                "cloudnet",
            ]
        )

        assert s.returncode == 0

    def test_cli_day_range(self):
        """
        Test orbital-radar for a range of days
        """

        s = subprocess.run(
            [
                "orbital-radar",
                "-g",
                "groundbased",
                "-n",
                "jue",
                "-s",
                "2021-04-01",
                "-e",
                "2021-04-01",
                "-f",
                "orbital_radar_config.toml",
                "-r",
                "mira",
                "-i",
                "cloudnet",
            ]
        )

        assert s.returncode == 0
