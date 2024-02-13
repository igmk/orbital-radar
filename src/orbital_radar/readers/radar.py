"""
This script contains all reader functions for the different radar formats. 
These functions are wrapped by the main function, which picks the correct
reader depending on the radar site.

The final output is always an xarray.Dataset with the two variables radar 
reflectivity "ze" in [mm6 m-3] and "vm" in [m s-1] as a function of range and
time.

The input Doppler velocity should be negative for downward motion and positive
for upward motion. This is changed to negative upward and positive downward to
match spaceborne convention.
"""

import os
import os.path
from glob import glob
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import xarray as xr

from orbital_radar.readers.cloudnet import read_cloudnet


class Radar:
    """
    This class selects the reading function for the provided site and
    performs quality-checks of the imported data. The output contains "ze"
    and "vm" variables.

    Implemented readers
    -------------------
    bco: Barbados Cloud Observatory, Barbados
    jue: JOYCE, Juelich, Germany
    mag: Magurele, Romania
    min: Mindelo, Cape Verde
    nor: Norunda, Sweden
    nya: Ny-Alesund, Svalbard
    mirac_p5: Polar 5, Mirac radar
    rasta: Falcon, RASTA radar
    arm: ARM sites
    pamtra: PAMTRA simulations
    cloudnet: cloudnet format
    """

    def __init__(self, date, site_name, path, input_radar_format) -> None:
        """
        Reads hourly radar data for a specific site and date with these
        standardized output variables:
        - Radar reflectivity (ze) in mm6 m-3
        - Mean Doppler velocity (vm) in m s-1
        - Range as height above NN (range) in m
        - Time (time)

        Parameters
        ----------
        date : pd.Timestamp
            Radar data will be read for this day
        site_name : str
            Name of the radar site (e.g. Mindelo)
        path : str
            Directory of radar data. The NetCDF files are expected inside a
            sub-folder structure starting from path: "path/yyyy/mm/dd/*.nc".
            Other options are not implemented yet.
        input_radar_format : str
            Format of input NetCDF radar data (uoc_v0, uoc_v1, uoc_v2,
            uoc_geoms, bco, mirac_p5). Default is cloudnet. Otherwise, these
            formats might be correct: uoc_v0 for nya, uoc_v1 for jue, uoc_v2
            for nor and mag, geoms for min, bco for bco, mirac_p5 for mp5.
        """

        self.date = pd.Timestamp(date)
        self.site_name = site_name
        self.path = path
        self.make_date_path()
        self.ds_rad = xr.Dataset()

        # defines the reader for each site
        readers = {
            "uoc_v0": self.read_uoc_v0,
            "uoc_v1": self.read_uoc_v1,
            "uoc_v2": self.read_uoc_v2,
            "geoms": self.read_geoms,
            "bco": self.read_bco,
            "mirac_p5": self.read_mirac_p5,
            "cloudnet": self.read_cloudnet,
            "pamtra": self.read_pamtra,
            "rasta": self.read_rasta,
            "arm": self.read_arm,
        }

        reader = readers.get(input_radar_format)
        if reader is None:
            raise NotImplementedError(
                f"No reader that handles input format {input_radar_format}. "
                f"Please choose one of {list(readers.keys())}."
            )
        reader()

        if self.ds_rad == xr.Dataset():
            print("No radar data found.")
            self.ds_rad = None

        else:
            print("Vm sign convention: negative=upward, " "positive=downward")

            self.ds_rad["vm"] = -self.ds_rad["vm"]

            print(f"Quality checks for {self.site_name} radar data.")

            # ensure ze and vm variables exist
            assert "ze" in list(self.ds_rad)
            assert "vm" in list(self.ds_rad)

            # ensure same dimension order
            if "height" in list(self.ds_rad.dims):
                dim_order = ["time", "height"]
            else:
                dim_order = ["time", "range"]
            self.ds_rad["ze"] = self.ds_rad.ze.transpose(*dim_order)
            self.ds_rad["vm"] = self.ds_rad.vm.transpose(*dim_order)

            # ensure reasonable value ranges
            assert (
                self.ds_rad.ze.isnull().all() or self.ds_rad.ze.min() >= 0
            ), "Ze out of range."
            assert self.ds_rad.ze.isnull().all() or (
                10 * np.log10(self.ds_rad.ze.max()) < 100
            ), "Ze out of range."

            assert (
                self.ds_rad.vm.isnull().all() or self.ds_rad.vm.min() > -80
            ), "Vm values out of range."
            assert (
                self.ds_rad.vm.isnull().all() or self.ds_rad.vm.max() < 80
            ), "Vm values out of range."

            # make sure that alt is in the data
            assert "alt" in list(self.ds_rad), "Altitude not found."

    def make_date_path(self):
        """
        Creates path with date structure if it exists. Otherwise, uses regular
        path without date extension.
        """

        date_path = os.path.join(
            self.path,
            self.date.strftime(r"%Y"),
            self.date.strftime(r"%m"),
            self.date.strftime(r"%d"),
        )

        if os.path.exists(date_path):
            self.path = date_path

        elif os.path.exists(self.path):
            pass  # use regular path without date extension

        else:
            raise FileNotFoundError(
                f"The radar data path {self.path} does not exist"
            )

    def get_all_files(self, pattern):
        """
        Lists all radar files in the directory.

        Parameters
        ----------
        pattern : str
            Specific file pattern depending on the product.

        Returns
        -------
        list
            list of all files inside the directory
        """

        pattern_path = os.path.join(self.path, pattern)
        files = sorted(glob(pattern_path))

        if len(files) == 0:
            Warning(f"No files found with pattern: {pattern_path}")

        return files

    @staticmethod
    def status_message(i, file, files):
        """
        This message will be printed while reading the radar data.
        """

        print(f"Reading radar file {i+1}/{len(files)}: {file}")

    @staticmethod
    def remove_duplicate_times(ds):
        """
        Removes duplicate times.

        Parameters
        ----------
        ds : xr.Dataset
            Any data with a "time" coordinate.
        """

        _, index = np.unique(ds["time"], return_index=True)
        ds = ds.isel(time=index)

        return ds

    def convert_and_sort_time(self, base_time):
        """
        Convert time in seconds since base to np.datetime64 format and sort
        time.

        Parameters
        ----------
        base_time : str
            Base time as string (e.g. "1970-01-01")
        """

        self.ds_rad["time"] = self.ds_rad["time"].astype(
            "timedelta64[s]"
        ) + np.datetime64(base_time)

        self.ds_rad = self.ds_rad.sel(time=np.sort(self.ds_rad.time))

        # ensure that time of files matches provided date
        assert np.abs(
            self.date.to_datetime64() - self.ds_rad.time
        ).max() < np.timedelta64(2, "D")

    def read_uoc_v2(self):
        """
        This function reads the radar netCDF files of the RPG w-band radar
        The data are precessed with the Matlab code of the Uuniversity of
        Cologne, UoC. They correspond to the level 2 version of the data
        processing

        Units of file:
        ze unit: mm6 m-3
        vm unit: m s-1

        Note: fill value -999 in attributes not recognized by xarray.
        """

        files = self.get_all_files("*compact_v2.nc")

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file, decode_times=False) as ds:
                ds.load()

            ds = self.remove_duplicate_times(ds)

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze", "vm"]], self.ds_rad], combine_attrs="override"
            )

            # extract instrument location and altitude
            self.ds_rad["lon"] = ds["lon"]
            self.ds_rad["lat"] = ds["lat"]
            self.ds_rad["alt"] = ds["zsl"]

        self.convert_and_sort_time(base_time="2001-01-01")

        # replace fill_value by nan
        self.ds_rad = self.ds_rad.where(self.ds_rad != -999)

    def read_uoc_v0(self):
        """
        This function reads the radar netCDF files of the RPG w-band radar
        The data are precessed with the Matlab code of the Uuniversity of
        Cologne, UoC. They correspond to the level 2 version of the data
        processing

        Units of file:
        ze unit: dBZ
        vm unit: m s-1

        Note: Only one file per day
        """

        files = self.get_all_files("*joyrad94_nya_lv1b_*")

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file, decode_times=False) as ds:
                ds.load()

            ds = ds.rename({"height": "range"})

            # round times to full seconds
            ds["time"] = np.around(ds["time"]).astype("int")

            ds = self.remove_duplicate_times(ds)

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze", "vm"]], self.ds_rad], combine_attrs="override"
            )

            # extract instrument location and altitude
            self.ds_rad["lon"] = ds["lon"]
            self.ds_rad["lat"] = ds["lat"]
            self.ds_rad["alt"] = ds["instrument_altitude"]

        self.convert_and_sort_time(base_time="2001-01-01")

        # convert from dB to linear units
        self.ds_rad["ze"] = 10 ** (0.1 * self.ds_rad["ze"])

    def read_uoc_v1(self):
        """
        This function reads the radar netCDF files of the RPG w-band radar
        The data are precessed with the Matlab code of the Uuniversity of
        Cologne, UoC. They correspond to the level 2 version of the data
        processing

        Units of file:
        ze unit: mm6 m-3
        vm unit: m s-1

        Note: Doppler spectra are not read to improve performance.
        """

        files = self.get_all_files("*ZEN_v2.nc")

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            xr_kwds = dict(drop_variables=["sze"], decode_times=False)
            with xr.open_dataset(file, **xr_kwds) as ds:
                ds.load()

            ds = self.remove_duplicate_times(ds)

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze", "vm"]], self.ds_rad], combine_attrs="override"
            )

            # extract instrument location and altitude
            self.ds_rad["lon"] = ds["lon"]
            self.ds_rad["lat"] = ds["lat"]
            self.ds_rad["alt"] = ds["zsl"]

        self.convert_and_sort_time(base_time="2001-01-01")

    def read_geoms(self):
        """
        This function reads the radar netCDF files of the RPG w-band radar
        The data are precessed with the Matlab code of the Uuniversity of
        Cologne, UoC. They correspond to the level 2 version of the data
        processing

        Units of file:
        ze unit: mm6 m-3
        vm unit: m s-1

        Note: fill value -999 in attributes not recognized by xarray.
        """

        files = self.get_all_files("*groundbased_radar_profiler*")

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file, decode_times=False) as ds:
                ds.load()

            ds = ds.rename(
                {
                    "RANGE": "range",
                    "DATETIME": "time",
                    "RADAR.REFLECTIVITY.FACTOR": "ze",
                    "DOPPLER.VELOCITY_MEAN": "vm",
                }
            )
            ds = self.remove_duplicate_times(ds)

            # add range and time as coordinates
            ds.coords["range"] = ds["range"]
            ds.coords["time"] = ds["time"]

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze", "vm"]], self.ds_rad], combine_attrs="override"
            )

            # extract instrument location and altitude
            self.ds_rad["lon"] = ds["LONGITUDE.INSTRUMENT"]
            self.ds_rad["lat"] = ds["LATITUDE.INSTRUMENT"]
            self.ds_rad["alt"] = ds["ALTITUDE.INSTRUMENT"]

        self.convert_and_sort_time(base_time="2001-01-01")

        # replace fill_value by nan
        self.ds_rad = self.ds_rad.where(self.ds_rad != -999)

    def read_bco(self):
        """
        This function reads the radar netCDF files of the RPG w-band radar
        The data are precessed with the Matlab code of the Uuniversity of
        Cologne, UoC. They correspond to the level 2 version of the data
        processing

        Units of file:
        ze unit: dBZ
        vm unit: m s-1

        Note: Only one file per day
        """

        files = self.get_all_files(f'*{self.date.strftime(r"%Y%m%d")}.nc')

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file, decode_times=False) as ds:
                ds.load()

            ds = ds.rename({"Ze": "ze", "VEL": "vm"})

            ds = self.remove_duplicate_times(ds)

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze", "vm"]], self.ds_rad], combine_attrs="override"
            )

            # extract instrument location and altitude
            self.ds_rad["lon"] = ds["lon"]
            self.ds_rad["lat"] = ds["lat"]
            self.ds_rad["alt"] = np.nan

        self.convert_and_sort_time(base_time="1970-01-01")

        # convert from dB to linear units
        self.ds_rad["ze"] = 10 ** (0.1 * self.ds_rad["ze"])

        # apply dBz threshold
        dbz_threshold = 10 ** (95.5 / 10.0)
        dbz_noise_level = self.ds_rad["range"] ** 2 / dbz_threshold
        dbz_noise_level = xr.DataArray(
            dbz_noise_level,
            dims="range",
            coords={"range": self.ds_rad["range"]},
        )
        self.ds_rad["ze"] = self.ds_rad["ze"].where(
            self.ds_rad["ze"] > dbz_noise_level
        )
        self.ds_rad["vm"] = self.ds_rad["vm"].where(
            self.ds_rad["ze"] > dbz_noise_level
        )

    def read_mirac_p5(self):
        """
        This function reads the radar netCDF files of the RPG W-band radar
        onboard the Polar 5 aircraft.

        Units of file:
        ze unit: dBZ

        Note: no mean Doppler velocity available.
        """

        files = self.get_all_files(f'*{self.date.strftime(r"%Y%m%d")}*.nc')

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file, decode_times=False) as ds:
                ds.load()

            ds = ds.rename({"Ze": "ze"})

            ds = self.remove_duplicate_times(ds)

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze"]], self.ds_rad], combine_attrs="override"
            )

            # extract instrument location and altitude
            self.ds_rad["lon"] = ds["lon"]
            self.ds_rad["lat"] = ds["lat"]
            self.ds_rad["alt"] = ds["alt"]

        self.convert_and_sort_time(base_time="2017-01-01")

        # replace fill_value by nan
        self.ds_rad = self.ds_rad.where(self.ds_rad != -999)

        # add dummy variable for Doppler velocity
        self.ds_rad["vm"] = xr.DataArray(
            np.zeros(self.ds_rad["ze"].shape),
            dims=["time", "height"],
            coords={
                "time": self.ds_rad["time"],
                "height": self.ds_rad["height"],
            },
        )

    def read_cloudnet(self):
        """
        Reads radar reflectivity and Doppler velocity from Cloudnet categorize
        files.

        Note: Cloudnet height is already in height above mean sea level.
        """

        ds = read_cloudnet(
            attenuation_correction_input="cloudnet_categorize",
            date=self.date,
            site_name=self.site_name,
            path=self.path,
            add_date=False,
        )

        ds = ds.rename({"Z": "ze", "v": "vm"})
        ds = self.remove_duplicate_times(ds)

        self.ds_rad = ds[["ze", "vm"]]

        # extract instrument location and altitude
        self.ds_rad["lon"] = ds["longitude"]
        self.ds_rad["lat"] = ds["latitude"]
        self.ds_rad["alt"] = ds["altitude"]

        # convert from dB to linear units
        self.ds_rad["ze"] = 10 ** (0.1 * self.ds_rad["ze"])

        # set inf ze to nan
        self.ds_rad["ze"] = self.ds_rad["ze"].where(
            self.ds_rad["ze"] != np.inf
        )

        # set very low vm to nan
        self.ds_rad["vm"] = self.ds_rad["vm"].where(self.ds_rad["vm"] > -500)

        self.ds_rad["vm"] = self.ds_rad["vm"].where(self.ds_rad["vm"] < 500)

    def read_pamtra(self):
        """
        Reads PAMTRA simulation for a point location as a function of time.

        Attenuation: The output radar reflectivity contains attenuation for
        bottom-up view (groundbased radar), and additionally one radar
        reflectivity without attenuation.

        Convention of output:
        ze: radar reflectivity without attenuation
        ze_top_down: radar reflectivity with attenuation for top-down view
        ze_bottom_up: radar reflectivity with attenuation for bottom-up view

        Units of file:
        ze unit: dBZ
        vm unit: m s-1
        """

        files = self.get_all_files(f'*{self.date.strftime(r"%Y%m%d")}*v1.nc')

        if len(files) == 0:
            files = self.get_all_files(f'*{self.date.strftime(r"%Y%m%d")}*v0.nc')

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file) as ds:
                ds.load()

            # currently, only 94 GHz simulations are supported
            assert ds.grid_y.size == 1
            ds = ds.isel(grid_y=0)

            # change heightbins to actual height values
            assert (ds.height.diff("grid_x") == 0).all()
            ds["height"] = ds["height"].isel(grid_x=0)
            ds = ds.swap_dims({"heightbins": "height"})

            # rename variables
            ds = ds.rename(
                {
                    "Ze": "ze",
                    "Radar_MeanDopplerVel": "vm",
                    "datatime": "time",
                    "longitude": "lon",
                    "latitude": "lat",
                }
            )

            assert ds.frequency == 94
            assert ds.radar_polarisation.size == 1
            assert ds.radar_peak_number.size == 1

            ds["ze"] = ds["ze"].isel(
                frequency=0, radar_polarisation=0, radar_peak_number=0
            )
            ds["vm"] = ds["vm"].isel(
                frequency=0, radar_polarisation=0, radar_peak_number=0
            )

            ds = ds.swap_dims({"grid_x": "time"})
            ds = ds.reset_coords()

            # check if attenuation correction was performed by pamtra
            props = {
                p.split(": ")[0]: p.split(": ")[1]
                for p in ds.attrs["properties"]
                .replace("'", "")[1:-1]
                .split(", ")
            }

            # two-way attenuation handling
            da_att = 2 * (
                ds.Attenuation_Hydrometeors.isel(frequency=0)
                + ds.Attenuation_Atmosphere.isel(frequency=0)
            )
            da_att_bu = da_att.cumsum("height")
            da_att_td = (
                da_att.sel(height=np.flip(ds.height))
                .cumsum("height")
                .sel(height=ds.height)
            )

            # radar reflectivity contains no attenuation
            if props["radar_attenuation"] == "disabled":
                pass

            # radar reflectivity contains attenuation for top-down view
            elif props["radar_attenuation"] == "top-down":
                ds["ze"] = ds["ze"] + da_att_td

            # radar reflectivity contains attenuation for bottom-up view
            elif props["radar_attenuation"] == "bottom-up":
                ds["ze"] = ds["ze"] + da_att_bu

            else:
                raise ValueError(
                    f"Attenuation correction {props['radar_attenuation']} not "
                    "supported."
                )

            ds["ze_bottom_up"] = ds["ze"] - da_att_bu
            ds["ze_top_down"] = ds["ze"] - da_att_td

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze", "vm", "ze_bottom_up", "ze_top_down"]], self.ds_rad],
                combine_attrs="override",
            )

            # add longitude and latitude
            assert (ds["lon"].diff("time") == 0).all()
            assert (ds["lat"].diff("time") == 0).all()

            self.ds_rad["lon"] = ds["lon"].isel(time=0).reset_coords(drop=True)
            self.ds_rad["lat"] = ds["lat"].isel(time=0).reset_coords(drop=True)

            if "alt" not in list(self.ds_rad):
                print("No altitude found in PAMTRA file. Setting alt to 0 m.")
                self.ds_rad["alt"] = 0

        # convert from dB to linear units
        self.ds_rad["ze"] = 10 ** (0.1 * self.ds_rad["ze"])
        self.ds_rad["ze_bottom_up"] = 10 ** (0.1 * self.ds_rad["ze_bottom_up"])
        self.ds_rad["ze_top_down"] = 10 ** (0.1 * self.ds_rad["ze_top_down"])

    def read_rasta(self):
        """
        This function reads the radar netCDF files of the airborne RASTA radar.
        The time-dependent height is interpolated onto a regular height grid.
        No correction for gas attenuation is applied (gaseous_twowayatt).
        The horizontal aircraft speed is also read.

        The following two flags are applied to ze and vm:

        First flag array: flag (1 and 2 are removed)
        -4: nadir not available
        -3: Last gates not valid
        -2: First gates not valid
        -1: beyond maximum range
        0: no cloud
        1: cloud or precipitation
        2: possible cloud
        3: ground echo
        4: ghost ground echo (downward domain)
        5: ghost ground echo (upward domain)
        6: underground signal
        7: underground noise
        8: Z can be interpolated

        Second flag array: flag_Z_interpolated (2 and 3 are removed)
        0: no interpolation
        1: interpolated but data available
        2: interpolated and no data available
        3: ghost ground echo (upward) is interpolated

        Units of file:
        ze unit: dBZ
        vm unit: m s-1
        ac_speed unit: m s-1

        Note: only the vertical-pointing antennas are used.
        """

        files = self.get_all_files(f'*{self.date.strftime(r"%Y%m%d")}*.nc')

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file) as ds:
                ds.load()

            ds = self.remove_duplicate_times(ds)

            # filter data
            keep_data = (
                (ds.flag != 1)
                & (ds.flag != 2)
                & (ds.flag_Z_interpolated != 2)
                & (ds.flag_Z_interpolated != 3)
            )
            ds["Z_interpolated"] = ds["Z_interpolated"].where(keep_data)
            ds["Vz"] = ds["Vz"].where(keep_data)

            # change doppler velocity from positive upward to negative upward
            ds["Vz"] = -ds["Vz"]

            # rename variables
            ds = ds.rename(
                {
                    "longitude": "lon",
                    "latitude": "lat",
                    "height": "height_index",  # this is not actual height
                    "aircraft_vh": "ac_speed",
                    "altitude": "alt",
                }
            )

            # interpolate onto regular height grid
            ds.coords["height"] = np.arange(-2000, 15000, 60)
            ze_height = np.zeros((len(ds["time"]), len(ds["height"])))
            vm_height = np.zeros((len(ds["time"]), len(ds["height"])))
            for i in range(len(ds.time)):

                # interpolate ze in linear space
                f_ze = interp1d(
                    ds["height_2D"].isel(time=i).values * 1e3,
                    10 ** (0.1 * ds["Z_interpolated"].isel(time=i).values),
                    kind="linear",
                    fill_value=np.nan,
                    bounds_error=False,
                )
                ze_height[i, :] = f_ze(ds["height"].values)

                # interpolate vm
                f_vm = interp1d(
                    ds["height_2D"].isel(time=i).values * 1e3,
                    ds["Vz"].isel(time=i).values,
                    kind="linear",
                    fill_value=np.nan,
                    bounds_error=False,
                )
                vm_height[i, :] = f_vm(ds["height"].values)

            # add variables to dataset and name them ze and vm
            ds["ze"] = xr.DataArray(
                ze_height,
                dims=["time", "height"],
                coords={"time": ds["time"], "height": ds["height"]},
            )
            ds["vm"] = xr.DataArray(
                vm_height,
                dims=["time", "height"],
                coords={"time": ds["time"], "height": ds["height"]},
            )

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [
                    ds[["ze", "vm", "lon", "lat", "alt", "ac_speed"]],
                    self.ds_rad,
                ],
                combine_attrs="override",
            )

            self.convert_and_sort_time(base_time=self.date)

    def read_arm(self):
        """
        Reader for ARM radar data. The reader uses the best estimate of the
        radar reflectivity.

        Units of file:
        ze unit: dBZ
        vm unit: m s-1
        """

        files = self.get_all_files(f'*{self.date.strftime(r"%Y%m%d")}*.cdf')

        if len(files) == 0:
            return None

        for i, file in enumerate(files):
            self.status_message(i, file, files)

            with xr.open_dataset(file) as ds:
                ds.load()

            ds = self.remove_duplicate_times(ds)

            ds = ds.rename(
                {
                    "mean_doppler_velocity": "vm",
                    "reflectivity_best_estimate": "ze",
                }
            )

            # override keeps attributes from the last file opened
            self.ds_rad = xr.merge(
                [ds[["ze", "vm"]], self.ds_rad], combine_attrs="override"
            )

            # add longitude and latitude
            self.ds_rad["lon"] = ds["lon"]
            self.ds_rad["lat"] = ds["lat"]
            self.ds_rad["alt"] = ds["alt"]

        self.ds_rad["ze"] = 10 ** (0.1 * self.ds_rad["ze"])
