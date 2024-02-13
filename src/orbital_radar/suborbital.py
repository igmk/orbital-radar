"""
This module contains the OrbitalRadar class that runs the simulator for
suborbital radar data. It is a subclass of the Simulator class.

Difference between ground-based and airborne radar geometry:
- along-track coordinate from mean wind for ground based and mean flight vel.
from airborne radar
- no ground echo added to airborne radar
- range grid of airborne radar assumed to be height above mean sea level and
height above ground for groundbased
- no attenuation correction for airborne radar
- lat/lon coordinates included as input to airborne radar
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from orbital_radar.helpers import db2li, li2db
from orbital_radar.radarspec import RadarBeam
from orbital_radar.readers.cloudnet import read_cloudnet
from orbital_radar.readers.config import read_config
from orbital_radar.readers.radar import Radar
from orbital_radar.simulator import Simulator
from orbital_radar.version import __version__
from orbital_radar.writers.spaceview import write_spaceview


class Suborbital(Simulator):
    """
    Run the simulator for suborbital radar data.
    """

    # list of all suborbital radar locations
    names = {
        "groundbased": [
            "bco",
            "jue",
            "nor",
            "mag",
            "min",
            "nya",
            "arm",
            "pamtra",
        ],
        "airborne": [
            "mp5",
            "rasta",
        ],
    }

    def __init__(
        self, geometry, name, config_file, suborbital_radar, input_radar_format
    ):
        """
        Initialize the simulator for suborbital radar data.

        Parameters
        ----------
        geometry : str
            Observation geometry of radar (groundbased or airborne).
        name : str
            Name of the suborbital radar (abbreviated).
        config_file : str
            Path to the configuration file that contains the site-dependent
            parameters and directory paths.
        suborbital_radar : str
            Name of the suborbital radar (abbreviated).
        input_radar_format : str
            Format of the input radar data (e.g. cloudnet).
        """

        # make sure that geometry is valid
        if geometry not in self.names.keys():
            raise ValueError(
                f"Geometry {geometry} not implemented. Choose from "
                f"{list(self.names.keys())}"
            )

        # check if site is in list of implemented sites
        if name not in self.names[geometry]:
            raise ValueError(
                f"Site {name} not implemented. Choose from "
                f"{self.names[geometry]}"
            )

        # check if config file is provided and exists
        if config_file is None:
            raise ValueError("No configuration file provided")

        if not os.path.isfile(
            os.path.join(os.environ["ORBITAL_RADAR_CONFIG_PATH"], config_file)
        ):
            raise FileNotFoundError(
                f"Configuration file {config_file} not found"
            )

        # set class attributes
        self.geometry = geometry
        self.name = name
        self.suborbital_radar = suborbital_radar
        self.input_radar_format = input_radar_format

        # attributes that will be derived
        self.is_sea_level = False

        # read configuration file
        self.config = read_config(config_file)

        # check if output path exists
        assert os.path.exists(
            self.config["paths"][self.name]["output"]
        ), f"Output path {self.config['paths'][self.name]['output']} does not exist"

        self.path_out = self.config["paths"][self.name]["output"]
        self.frequency = self.config["suborbital_radar"][
            self.suborbital_radar
        ]["frequency"]

        # preparation of input radar data
        self.prepare = self.config["prepare"]["general"]
        self.prepare.update(self.config["prepare"][self.geometry])

        # overview of simulation settings
        self.summary

        # initialize simulator class with spaceborne radar settings
        super().__init__(
            sat_name=self.config["spaceborne_radar"]["sat_name"],
            file_earthcare=self.config["spaceborne_radar"]["file_earthcare"],
            nyquist_from_prf=self.config["spaceborne_radar"]["nyquist_from_prf"],
            ms_threshold=self.config["spaceborne_radar"]["ms_threshold"],
            ms_threshold_integral=self.config["spaceborne_radar"][
                "ms_threshold_integral"
            ],
            **self.config["spaceborne_radar"]["radar_specs"],
        )

    @property
    def summary(self):
        """
        Prints short summary of simulator settings.
        """

        print(f"Site: {self.name}")
        print("\n")

        print("Directory paths:")
        print(f"Input: {self.config['paths'][self.name]['radar']}")
        print(f"Output: {self.config['paths'][self.name]['output']}")
        print("\n")

        if self.geometry == "groundbased":
            print("Groundbased data is prepared with:")
            print(f"Mean wind: {self.prepare['mean_wind']} m/s")

        if self.geometry == "airborne":
            print("Airborne data is prepared with:")
            print(
                f"Mean flight velocity: "
                f"{self.prepare['mean_flight_velocity']} m/s"
            )

        print(f"Height min: {self.prepare['height_min']} m")
        print(f"Height max: {self.prepare['height_max']} m")
        print(f"Height res: {self.prepare['height_res']} m")
        print("\n")

    @staticmethod
    def prepare_dates(start_date, end_date):
        """
        Creates a date array from start to end date.

        Parameters
        ----------
        start_date : np.datetime64
            Start date.
        end_date : np.datetime64
            End date.

        Returns
        -------
        dates: pd.DatetimeIndex
            Date range from start to end date.
        """

        # check if start date is before end date
        if start_date > end_date:
            raise ValueError("Start date must be before end date")

        dates = pd.date_range(start_date, end_date)

        return dates

    def convert_frequency(self, ds):
        """
        Convert frequency from 35 to 94 GHz.

        The conversion is based on Kollias et al. (2019)
        (doi: https://doi.org/10.5194/amt-12-4949-2019)

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" variable in mm6/mm3. Ze was measured at 35 GHz.

        Returns
        -------
        ds : xarray.Dataset
            Data with converted "ze" variable. Ze is now transformed to 94 GHz.
        """

        # keep only reflectivities below 30 dBZ
        ds["ze"] = ds["ze"].where(ds["ze"] < db2li(30))

        a = -16.8251
        b = 8.4923
        ds["ze"] = db2li(
            li2db(ds["ze"]) - 10**a * (li2db(ds["ze"]) + 100) ** b
        )

        # set negative ze to zero
        ds["ze"] = ds["ze"].where(ds["ze"] > 0.0, 0.0)

        return ds

    def correct_dielectric_constant(self, ds):
        r"""
        Apply correction for dielectric constant assumed in Ze calculation
        of suborbital radar to match the dielectric constant of the
        spaceborne radar.

        Correction equation with :math:`K_g` and :math:`K_s` as dielectric
        constants of the suborbital and spaceborne radar, respectively:

        .. math::
           Z_e = 10 \log_{10} \left( \frac{K_s}{K_g} \right) + Z_e

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" variable.
        """

        correction = (
            self.config["spaceborne_radar"]["k2"]
            / self.config["suborbital_radar"][self.suborbital_radar]["k2"]
        )

        ds["ze"] = db2li(li2db(ds["ze"]) + 10 * np.log10(correction))

        return ds

    def add_vmze_attrs(self, ds):
        """
        Adds attributes to Doppler velocity and radar reflectivity variables.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" and "vm" variables.

        Returns
        -------
        ds : xarray.Dataset
            Data with added attributes.
        """

        ds["ze"].attrs = dict(
            units="mm6 m-3",
            standard_name="radar_reflectivity",
            long_name="Radar reflectivity",
            description="Radar reflectivity",
        )

        ds["vm"].attrs = dict(
            units="m s-1",
            standard_name="Doppler_velocity",
            long_name="Doppler velocity",
            description="Doppler velocity",
        )

        return ds

    def check_is_sea_level(self, ds):
        """
        Check if input radar range/height grid is defined with respect to
        ground level or sea level. The input to the simulator should be wrt.
        sea level.

        Parameters
        ----------
        ds : xr.Dataset
            Input radar data
        """

        if "height" in list(ds.coords.keys()):
            self.is_sea_level = True
        
        else:
            self.is_sea_level = False

    def range_to_height(self, ds):
        """
        Convert range coordinate to height coordinate by adding the station
        height above mean sea level to the range coordinate.

        The altitude is pre-defined for each station in the configuration file.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "range" coordinate.
        """

        ds["height"] = ds["range"] + ds.alt.item()

        # swap range with height
        ds = ds.swap_dims({"range": "height"})

        # drop range coordinate
        ds = ds.reset_coords(drop=True)

        return ds

    def create_along_track(self, ds):
        """
        Creates along-track coordinates from time coordinates.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "time" and "height" coordinates.

        Returns
        -------
        ds : xarray.Dataset
            Data with "along_track" coordinate.
        """

        if self.geometry == "groundbased":
            print("Using mean wind for along-track coordinates")
            v = self.prepare["mean_wind"]

        elif self.geometry == "airborne" and "ac_speed" in list(ds):
            print("Using flight velocity for along-track coordinates")
            v = ds.ac_speed.values

        elif self.geometry == "airborne" and "ac_speed" not in list(ds):
            print("Using mean flight velocity for along-track coordinates")
            v = self.prepare["mean_flight_velocity"]

        else:
            raise ValueError(
                f"Geometry {self.geometry} not implemented. Choose from "
                f"{list(self.names.keys())}"
            )

        # calculate the along-track distance
        dt = ds.time.diff("time") / np.timedelta64(1, "s")
        dt = xr.align(ds.time, dt, join="outer")[1].fillna(0)  # start is dt=0
        arr_along_track = np.cumsum(v * dt)

        da_along_track = xr.DataArray(
            arr_along_track, dims="time", coords=[ds.time], name="along_track"
        )
        da_along_track.attrs = dict(
            standard_name="along_track_distance",
            long_name="Along track distance",
            units="m",
            description="Distance along track of the suborbital radar",
        )

        # swap from time to along track
        ds = ds.assign_coords(along_track=da_along_track)
        ds = ds.swap_dims({"time": "along_track"})

        # add time as variable
        ds = ds.reset_coords()

        return ds

    def create_regular_height(self):
        """
        Creates regular height coordinate for suborbital radar.

        Returns
        -------
        xarray.DataArray
            Regular height coordinate for suborbital radar.
        """

        height_regular = np.arange(
            self.prepare["height_min"],
            self.prepare["height_max"],
            self.prepare["height_res"],
        )

        da_height_regular = xr.DataArray(
            height_regular, dims="height", coords=[height_regular]
        )
        da_height_regular.attrs = dict(
            units="m",
            standard_name="height",
            long_name="Height of radar bin above sea level",
            description="Height of radar bin above sea level",
        )

        return da_height_regular

    def create_regular_along_track(self, ds):
        """
        Creates regular along-track coordinate for suborbital radar.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "along_track" coordinate.

        Returns
        -------
        xarray.DataArray
            Regular along-track coordinate for suborbital radar.
        """

        along_track_res = np.round(
            ds.along_track.diff("along_track").median().item()
        )
        along_track_max = ds.along_track.max().item()

        along_track_regular = np.arange(
            0,
            along_track_max,
            along_track_res,
        )

        da_along_track_regular = xr.DataArray(
            along_track_regular,
            dims="along_track",
            coords=[along_track_regular],
        )
        da_along_track_regular.attrs = ds.along_track.attrs

        return da_along_track_regular

    def interpolate_to_regular_grid(self, ds):
        """
        Interpolates radar data to regular grid in along-track and height.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "time" and "height" coordinates.

        Returns
        -------
        ds : xarray.Dataset
            Data with interpolated "along_track" and "height" coordinates.
        """

        da_height_regular = self.create_regular_height()
        da_along_track_regular = self.create_regular_along_track(ds=ds)

        # interpolation along-track and height
        # workaround for time: convert to seconds since start, then
        # interpolate, then convert back to datetime
        da_time = ds["time"]
        ds = ds.interp(
            along_track=da_along_track_regular,
            height=da_height_regular,
            method="nearest",
        )
        # get nearest time for each regular along track grid point
        ds.coords["time"] = (
            ("along_track"),
            da_time.sel(along_track=ds.along_track, method="nearest").values,
        )

        # add attributes
        ds = self.add_vmze_attrs(ds)

        return ds

    def add_ground_echo(self, ds):
        """
        Calculates artificial ground echo inside ground-based radar range
        grid. The values are chosen such that the final ground echo after
        the along-range convolution is equal to the ground echo of the
        satellite. The pulse length used here is not the same as the pulse
        length of the satellite.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" and "vm" variables.

        Returns
        -------
        ds : xarray.Dataset
            Data with added ground echo.
        """

        assert len(np.unique(np.diff(ds.height))) == 1, (
            "Height grid is not equidistant. "
            "Range weighting function cannot be calculated."
        )

        # grid with size of two pulse lengths centered around zero
        height_bins = np.arange(
            -self.prepare["ground_echo_pulse_length"],
            self.prepare["ground_echo_pulse_length"]
            + self.prepare["height_res"],
            self.prepare["height_res"],
        )

        # calculate range weighting function
        weights = RadarBeam.normalized_range_weighting_function_default(
            pulse_length=self.prepare["ground_echo_pulse_length"],
            range_bins=height_bins,
        )

        ground_echo = weights * db2li(self.prepare["ground_echo_ze_max"])

        # add ground echo to dataset shifted by one height bin to have maximum
        # below zero
        # get closest height bin to ground
        idx = (np.abs(ds.height - ds.alt.item())).argmin()
        base = ds.height[idx].item()

        # insert half of the calculated ground echo and shift maximum below the
        # surface
        ground_echo = ground_echo[int(len(ground_echo) / 2) :]
        height_bins = (
            base
            + height_bins[int(len(height_bins) / 2) :]
            - self.prepare["height_res"]
        )

        # add ground echo to dataset (first fill nan values with zero in this
        # height interval)
        ds["ze"].loc[{"height": height_bins}] = (
            ds["ze"].loc[{"height": height_bins}].fillna(0)
        )
        ds["ze"].loc[{"height": height_bins}] += ground_echo

        return ds

    def add_groundbased_variables(self):
        """
        Add variables specific to groundbased simulator to the dataset, i.e.,
        the mean horizontal wind.
        """

        self.ds["mean_wind"] = xr.DataArray(
            self.prepare["mean_wind"],
            attrs=dict(
                standard_name="v_hor",
                long_name="Mean horizontal wind",
                units="m s-1",
                description="Mean horizontal wind",
            ),
        )

    def add_airborne_variables(self):
        """
        Add variables specific to airborne simulator to the dataset, i.e.,
        the mean flight velocity.
        """

        self.ds["mean_flight_velocity"] = xr.DataArray(
            self.prepare["mean_flight_velocity"],
            attrs=dict(
                standard_name="v_hor",
                long_name="Mean flight velocity",
                units="m s-1",
                description="Mean flight velocity",
            ),
        )

    def to_netcdf(self, date):
        """
        Writes dataset to netcdf file. Note that not all variables are stored.

        Parameters
        ----------
        date : np.datetime64
            Date of the simulation. Used to create the filename.
        """

        output_variables = [
            "sat_ifov",
            "sat_range_resolution",
            "sat_along_track_resolution",
            "ze",
            "vm",
            "ze_sat",
            "vm_sat",
            "vm_sat_vel",
            "ze_sat_noise",
            "vm_sat_noise",
            "vm_sat_folded",
            "nubf_flag",
            "ms_flag",
            "folding_flag",
        ]

        if self.geometry == "groundbased":
            output_variables += ["mean_wind"]

        if self.geometry == "airborne":
            output_variables += ["mean_flight_velocity"]

        # name of output nc file
        filename = (
            "_".join(
                [
                    "ora",
                    __version__,
                    self.config["spaceborne_radar"]["sat_name"],
                    "l1",
                    self.geometry,
                    self.name,
                    self.suborbital_radar,
                    pd.Timestamp(date).strftime("%Y%m%d") + "T000000",
                    pd.Timestamp(date).strftime("%Y%m%d") + "T235959",
                ]
            )
            + ".nc"
        )

        filename = os.path.join(
            self.path_out,
            filename,
        )

        write_spaceview(ds=self.ds[output_variables], filename=filename)

    def add_attenuation(self, ds, da_gas_atten):
        """
        Add attenuation to dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" variable. Unit: mm6 m-3
        da_gas_atten : xarray.DataArray
            Interpolated gas attenuation data on the same grid as ds.
            Unit: dBZ.

        Returns
        -------
        ds : xarray.Dataset
            Data with added attenuation.
        """

        # add attenuation in dB and convert back to
        ds["ze"] = db2li(li2db(ds["ze"]) + da_gas_atten)

        return ds

    def attenuation_correction(self, ds, ds_cloudnet):
        """
        Gas attenuation correction based on Cloudnet.

        Cloudnet contains gas attenuation (gas_atten) as a function of 137
        ERA5 levels. The height of each level varies with time. Therefore,
        the height is first interpolated onto the height grid of the radar
        data. Then, the gas attenuation is interpolated onto the time and
        height grid of the radar data. Finally, the gas attenuation is added
        to the radar reflectivity.

        There are major differences between the cloudnet_ecmwf and
        cloudnet_categorize attenuation products:
        - ecmwf height is time-dependent, categorize height is not
        - ecmwf height is wrt ground, categorize height is wrt mean sea level
        - ecmwf variable is named "radar_gas_atten", categorize variable is
        named "gas_atten"
        - ecmwf is calculated for both frequencies, categorize only for 94 GHz

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" variable. Unit: mm6 m-3
        ds_cloudnet : xarray.Dataset
            Cloudnet data with "gas_atten" variable. Unit: dBZ

        Returns
        -------
        ds : xarray.Dataset
            Data with added attenuation.
        """

        # select 94 GHz frequency
        if "frequency" in list(ds_cloudnet.dims):
            ds_cloudnet = ds_cloudnet.sel(frequency=94, method="nearest")

        if self.prepare["attenuation_correction_input"] == "cloudnet_ecmwf":
            lst_da_gas_atten = []
            for time in ds_cloudnet.time:

                # interpolate gas attenuation to radar height grid
                ds_cloudnet_t = ds_cloudnet.sel(time=time).swap_dims(
                    {"level": "height"}
                )

                da_cloudnet_gas_atten_t = ds_cloudnet_t.gas_atten.interp(
                    height=ds.height, method="linear"
                )

                lst_da_gas_atten.append(da_cloudnet_gas_atten_t)

            # merge all time steps
            da_gas_atten = xr.concat(lst_da_gas_atten, dim="time")

            # drop levels
            da_gas_atten = da_gas_atten.reset_coords(drop=True)

        elif (
            self.prepare["attenuation_correction_input"]
            == "cloudnet_categorize"
        ):
            # rename to match ecmwf variable
            ds_cloudnet = ds_cloudnet.rename({"radar_gas_atten": "gas_atten"})

            # interpolate to radar height grid
            da_gas_atten = ds_cloudnet.gas_atten.interp(
                height=ds.height, method="linear"
            )

        else:
            raise ValueError(
                f"Attenuation correction input "
                f"{self.prepare['attenuation_correction_input']} not "
                f"implemented"
            )

        # interpolate to radar time grid and extrapolate if needed
        da_gas_atten = da_gas_atten.interp(
            time=ds.time, method="linear", kwargs={"fill_value": "extrapolate"}
        )

        # ensures every time step contains attenuation in range column
        has_atten = ~da_gas_atten.isnull().all("height")
        assert (
            has_atten.all()
        ), f"Attenuation missing for times: {da_gas_atten.time[~has_atten]}"

        # fill nan values with zero
        da_gas_atten = da_gas_atten.fillna(0)

        # apply attenuation correction
        ds = self.add_attenuation(ds=ds, da_gas_atten=da_gas_atten)

        return ds

    def run_date(self, date, write_output=True):
        """
        Runs simulation for a single day.

        Parameters
        ----------
        date : np.datetime64
            Date to simulate.
        write_output : bool
            If True, write output to netcdf file.
        """

        # read radar data
        if self.input_radar_format == "cloudnet":
            radar_path = self.config["paths"][self.name]["cloudnet"]
        else:
            radar_path = self.config["paths"][self.name]["radar"]

        rad = Radar(
            date=date,
            site_name=self.name,
            path=radar_path,
            input_radar_format=self.input_radar_format,
        )

        # skip if radar data does not exist
        if rad.ds_rad is None:
            print(f"{date}: No radar data found")
            return

        # read cloudnet data
        if self.geometry == "groundbased":
            ds_cloudnet = read_cloudnet(
                attenuation_correction_input=self.prepare[
                    "attenuation_correction_input"
                ],
                date=date,
                site_name=self.name,
                path=self.config["paths"][self.name]["cloudnet"],
            )

            if ds_cloudnet is None:
                self.prepare["attenuation_correction"] = False

        # frequency conversion
        if self.frequency == 35:
            rad.ds_rad = self.convert_frequency(rad.ds_rad)

        # correct dielectric constant
        if self.frequency == 94:
            rad.ds_rad = self.correct_dielectric_constant(rad.ds_rad)

        # range to height
        self.check_is_sea_level(rad.ds_rad)

        if (self.geometry == "groundbased") and not self.is_sea_level:
            print("Converting radar range grid to height above mean sea level")
            rad.ds_rad = self.range_to_height(rad.ds_rad)
        else:
            print(
                "Assume that input radar grid is defined as height above mean "
                "sea level"
            )

        # create along track dimension
        rad.ds_rad = self.create_along_track(ds=rad.ds_rad)

        # interpolate to regular grid
        ds = self.interpolate_to_regular_grid(rad.ds_rad)

        if self.geometry == "groundbased":
            # attenuation correction
            # no attenuation correction with 35 GHz radar reflectivity
            if self.frequency == 35:
                self.prepare["attenuation_correction"] = False

            if self.prepare["attenuation_correction"]:
                ds = self.attenuation_correction(ds, ds_cloudnet)

            # add ground echo
            ds = self.add_ground_echo(ds)

        # run simulator
        self.transform(ds)

        # add horizontal wind variable
        if self.geometry == "groundbased":
            self.add_groundbased_variables()

        if self.geometry == "airborne":
            self.add_airborne_variables()

        # write output to file
        if write_output:
            self.to_netcdf(date=date)

    def run(self, start_date, end_date, write_output=True):
        """
        Runs simulation for all days in the time frame.

        Parameters
        ----------
        start_date : np.datetime64
            Start date.
        end_date : np.datetime64
            End date (inclusive).
        """

        dates = self.prepare_dates(start_date, end_date)

        for i, date in enumerate(dates):
            print(f"Processing {date} ({i+1}/{len(dates)})")

            self.run_date(date, write_output=write_output)
