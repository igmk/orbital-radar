"""
Runs the orbital radar simulator.
"""

import numpy as np
import xarray as xr
from scipy import stats
from scipy.interpolate import interp1d

from orbital_radar.helpers import db2li, li2db
from orbital_radar.plotting.curtains import plot_along_track
from orbital_radar.plotting.histogram import plot_histogram
from orbital_radar.plotting.scatter import plot_scatter
from orbital_radar.radarspec import RadarBeam
from orbital_radar.version import __version__


class Simulator:
    """
    Runs the orbital radar simulator.
    """

    def __init__(
        self,
        sat_name,
        file_earthcare=None,
        nyquist_from_prf=False,
        ms_threshold=12,
        ms_threshold_integral=41,
        **radar_specs,
    ):
        """
        Initialize the simulator class. The input dataset will be extended with
        intermediate simulation steps.

        To run the simulator:
        - initialize the class
        - run the transform method

        Requirement:
        - all nan values should be filled with zeros
        - along-track and height coordinates should be monotonic increasing,
          evenly spaced, and multiples of the satellite resolution to ensure
          that each satellite bin contains the same number of high-resolution
          bins (e.g. 0, 100, 200, 300... --> 0, 500, 1000)

        Parameters
        ----------
        sat_name : str
            Name of the satellite. This is used to get the radar specifications
            from the config file.
        nyquist_from_prf : bool
            If True, the Nyquist velocity is calculated from the pulse
            repetition frequency (PRF).
        file_earthcare : str
            path to file containing EarthCARE CPR weighting function. This
            file is used if the satellite name is 'earthcare'.
        radar_specs : dict
            Dictionary with radar specifications.
        """

        # initialize class variables
        self.sat_name = sat_name
        self.radar_specs = radar_specs
        self.ds = xr.Dataset()

        self.ms_threshold = ms_threshold
        self.ms_threshold_integral = ms_threshold_integral

        # get radar specifications
        self.beam = RadarBeam(
            sat_name=self.sat_name,
            file_earthcare=file_earthcare,
            nyquist_from_prf=nyquist_from_prf,
            **self.radar_specs,
        )

    def check_input_dataset(self):
        """
        Check user input for consistency.
        """

        # make sure that ds is an xarray dataset
        assert isinstance(self.ds, xr.Dataset)

        # make sure that dimensions are named correctly
        assert self.ds["ze"].dims == ("along_track", "height")

        # make sure that variables exist
        assert "ze" in self.ds

        # make sure that radar reflectivity is in linear units
        assert self.ds["ze"].min() >= 0

        # check if satellite resolution is a multiple of the range resolution
        assert (
            self.beam.spec.range_resolution
            % self.ds["height"].diff("height")[0]
            == 0
        ), (
            f"Height resolution is not a multiple of the satellite resolution: "
            f"{self.ds['height'].diff('height')[0]} m"
        )

        # check if range resolution is smaller or equal to satellite resolution
        assert (
            self.ds["height"].diff("height")[0]
            <= self.beam.spec.range_resolution
        ), (
            f"Range resolution is larger than the satellite resolution: "
            f"{self.ds['height'].diff('height')[0]} m"
        )

    def prepare_input_dataset(self):
        """
        Prepares input dataset for computations. This only includes replacing
        nan values by zero in both ze and vm.
        """

        self.ds = self.ds.fillna(0)

        # make sure that ze has no nan values
        assert not self.ds["ze"].isnull().any()

        # make sure that vm has no nan values
        assert not self.ds["vm"].isnull().any()

    def calculate_along_track_sat_bin_edges(self):
        """
        Calculate the bin edges of the along-track satellite grid. This way
        is equivalent to height.
        """

        along_track_sat_bin_edges = np.append(
            self.ds["along_track_sat"]
            - self.beam.spec.along_track_resolution / 2,
            self.ds["along_track_sat"][-1]
            + self.beam.spec.along_track_resolution / 2,
        )

        return along_track_sat_bin_edges

    def calculate_height_sat_bin_edges(self):
        """
        Calculate the bin edges of the height satellite grid. This way is
        equivalent to along-track.
        """

        height_sat_bin_edges = np.append(
            self.ds["height_sat"] - self.beam.spec.range_resolution / 2,
            self.ds["height_sat"][-1] + self.beam.spec.range_resolution / 2,
        )

        return height_sat_bin_edges

    def convolve_along_track(self):
        """
        Calculates the along-track convolution from the input suborbital data
        using the along-track weighting function of the spaceborne radar.
        Further, the function calculates the error due to satellite velocity.
        """

        ds = self.ds[["ze", "vm"]].copy()
        ds = ds.fillna(0)

        # create dask array by splitting height to reduce memory when expanding
        # window dimension
        ds = ds.chunk(chunks={"height": 50})

        # create new dataset with window dimension stacked as third dimension
        weight = xr.DataArray(self.beam.atrack_weights, dims=["window"])
        ds = ds.rolling(along_track=len(weight), center=True).construct(
            "window"
        )

        # add error due to satellite motion to each doppler velocity window
        da_vel_error = xr.DataArray(self.beam.velocity_error, dims=["window"])
        ds["vm_err"] = ds["vm"] + da_vel_error
        ds["vm_err"] = ds["vm_err"].where(ds["vm"] != 0, ds["vm"])

        # calculate along-track convolution and convert dask to xarray
        self.ds["ze_acon"] = ds["ze"].dot(weight).compute()
        self.ds["vm_acon"] = ds["vm"].dot(weight).compute()
        self.ds["vm_acon_err"] = ds["vm_err"].dot(weight).compute()

    def integrate_along_track(self):
        """
        Integrates the along-track convoluted data to profiles, which represent
        the satellite's footprint. The along-track integration is given by the
        along track resolution satellite variable.

        Along track bins of satellite refer to center of field of view.
        """

        # create bin edges for along-track integration
        # the last radar bin is created only if it is included in the input
        # grid. the same convention is applied to the height grid
        along_track_sat_edges = np.arange(
            self.ds["along_track"][0],
            self.ds["along_track"][-1],
            self.beam.spec.along_track_resolution,
        )

        # create bin centers for along-track integration
        along_track_bin_center = (
            along_track_sat_edges[:-1] + along_track_sat_edges[1:]
        ) / 2

        # along-track integration onto satellite along-track grid
        kwds = {
            "group": "along_track",
            "bins": along_track_sat_edges,
            "labels": along_track_bin_center,
        }

        self.ds["ze_aconint"] = self.ds["ze_acon"].groupby_bins(**kwds).mean()
        self.ds["vm_aconint"] = self.ds["vm_acon"].groupby_bins(**kwds).mean()
        self.ds["vm_aconint_err"] = (
            self.ds["vm_acon_err"].groupby_bins(**kwds).mean()
        )

        # rename along-track dimension
        self.ds = self.ds.rename({"along_track_bins": "along_track_sat"})

    def convolve_height(self):
        """
        Convolution of the along-track integrated data with the range
        weighting function of the spaceborne radar.
        """

        # this defines the weights for the range gates that will be averaged
        da_range_weights = xr.DataArray(
            data=self.beam.range_weights,
            coords={"pulse_center_distance": self.beam.range_bins},
            dims=["pulse_center_distance"],
            name="range_weights",
        )

        # this defines the rolling window interval, i.e., the factor by which
        # the along-track resolution is reduced
        stride = int(
            self.beam.spec.range_resolution
            / self.ds["height"].diff("height")[0]
        )

        # create new dimension with all range gates that contribute to the
        # along-height convolution at each range gate
        ds = (
            self.ds[["ze_aconint", "vm_aconint", "vm_aconint_err"]]
            .rolling(height=len(da_range_weights), center=True)
            .construct("pulse_center_distance", stride=stride)
        )
        ds = ds.rename({"height": "height_sat"})

        # calculate along-range convolution
        self.ds["ze_sat"] = ds["ze_aconint"].dot(da_range_weights)
        self.ds["vm_sat"] = ds["vm_aconint"].dot(da_range_weights)
        self.ds["vm_sat_vel"] = ds["vm_aconint_err"].dot(da_range_weights)

    def calculate_nubf(self):
        r"""
        Calculates the non-uniform beam filling from the standard
        deviation of Ze within the radar volume.

        Currently, the flag is expressed as standard deviation only and no
        threshold to indicate high standard deviation is applied. This may
        be added in the future to reduce the output file size.
        """

        # create labels for each satellite pixel (height_sat x along_track_sat)
        labels = np.arange(
            self.ds["height_sat"].size * self.ds["along_track_sat"].size
        ).reshape(self.ds["ze_sat"].shape)

        # calculate bin edges of satellite grid
        along_track_sat_bin_edges = self.calculate_along_track_sat_bin_edges()
        height_sat_bin_edges = self.calculate_height_sat_bin_edges()

        # assign satellite pixel label to each input pixel of suborbital radar
        ix_along_track = np.searchsorted(
            along_track_sat_bin_edges[:-1],
            self.ds["along_track"].values,
            side="left",
        )
        ix_height = np.searchsorted(
            height_sat_bin_edges[:-1],
            self.ds["height"].values,
            side="left",
        )

        # adjust index at first position
        ix_height[ix_height == 0] = 1
        ix_along_track[ix_along_track == 0] = 1

        ix_height = ix_height - 1
        ix_along_track = ix_along_track - 1

        ix_height, ix_along_track = np.meshgrid(
            ix_along_track, ix_height, indexing="ij"
        )
        labels_input_grid = labels[ix_height, ix_along_track]

        # calculate standard deviation of ze on input grid in linear units
        # this is done with pandas for faster performance
        df_ze = (
            self.ds["ze"]
            .stack({"x": ("along_track", "height")})
            .to_dataframe()
        )
        df_ze["labels"] = labels_input_grid.flatten()
        df_nubf = li2db(df_ze["ze"]).groupby(df_ze["labels"]).std()

        # convert to xarray
        self.ds["nubf"] = xr.DataArray(
            df_nubf.values.reshape(labels.shape),
            dims=["along_track_sat", "height_sat"],
            coords={
                "along_track_sat": self.ds["along_track_sat"],
                "height_sat": self.ds["height_sat"],
            },
        )

    def calculate_nubf_flag(self, threshold=1):
        """
        Calculate non-uniform beam filling flag. The flag is 1 if the
        non-uniform beam filling is higher than a certain threshold, and 0
        otherwise.

        Parameters
        ----------
        threshold : float
            Threshold for non-uniform beam filling. The default is 1 dB.
        """

        self.ds["nubf_flag"] = (self.ds["nubf"] > threshold).astype("int")

    def calculate_vm_bias(self):
        """
        Calculate the satellite Doppler velocity bias between the estimate
        with and without satellite motion error.
        """

        self.ds["vm_bias"] = self.ds["vm_sat"] - self.ds["vm_sat_vel"]

    def calculate_vm_bias_flag(self, threshold=0.5):
        """
        Calculate the satellite Doppler velocity bias flag. The flag is 1 if
        the absolute satellite Doppler velocity bias is higher than 0.5 m s-1,
        and 0 otherwise.

        Parameters
        ----------
        threshold : float
            Threshold for satellite Doppler velocity bias. The default is 0.5
            m s-1.
        """

        self.ds["vm_bias_flag"] = (
            np.abs(self.ds["vm_bias"]) > threshold
        ).astype("int")

    def calculate_signal_fraction(self):
        """
        Calculates the fraction of bins that contain a ze signal above the
        detection limit of the spaceborne radar. The fraction is 1 if all
        bins contain signal, and 0 if no bins contain signal.
        """

        # calculate bin edges of satellite grid
        along_track_sat_bin_edges = self.calculate_along_track_sat_bin_edges()
        height_sat_bin_edges = self.calculate_height_sat_bin_edges()

        # calculate fraction of bins that contain signal
        self.ds["signal_fraction"] = self.ds["ze"] > 0

        self.ds["signal_fraction"] = (
            self.ds["signal_fraction"]
            .groupby_bins(
                "along_track",
                bins=along_track_sat_bin_edges,
                labels=self.ds["along_track_sat"].values,
            )
            .mean()
        ).rename({"along_track_bins": "along_track_sat"})

        self.ds["signal_fraction"] = (
            self.ds["signal_fraction"]
            .groupby_bins(
                "height",
                bins=height_sat_bin_edges,
                labels=self.ds["height_sat"].values,
            )
            .mean()
        ).rename({"height_bins": "height_sat"})

    def calculate_ms_flag(self):
        """
        Calculates the multiple scattering flag. The flag is 1 if multiple
        scattering occurs, and 0 if no multiple scattering occurs.

        The flag is calculated from the radar reflectivity of the spaceborne
        radar from these steps:
        - Calculate integral of radar reflectivity above a certain threshold
        from the top of the atmosphere (TOA) down to the surface.
        - Multiple scattering occurs if the integral reaches a critical value
        at a certain height.
        """

        # get ze above multiple scattering threshold
        da_ze_above_threshold = self.ds["ze_sat"] > db2li(self.ms_threshold)

        # integrate from top to bottom (this requires sel)
        self.ds["ms_flag"] = (
            self.ds["ze_sat"]
            .where(da_ze_above_threshold)
            .sel(height_sat=self.ds["height_sat"][::-1])
            .cumsum("height_sat")
            .sel(height_sat=self.ds["height_sat"])
        ) * self.beam.spec.range_resolution

        # convert to dBZ and calculate flag
        self.ds["ms_flag"] = (
            li2db(self.ds["ms_flag"]) > self.ms_threshold_integral
        ).astype("int")

        # set flag to 0 below the surface
        subsurface = self.ds["height_sat"].where(
            self.ds["height_sat"] < 0, drop=True
        )
        self.ds["ms_flag"].loc[{"height_sat": subsurface}] = 0

    def apply_detection_limit(self, var_ze, var_other: list):
        """
        Applies the detection limit of the spaceborne radar to the along-height
        convoluted data.

        Parameters
        ----------
        var_ze : xr.DataArray
            Radar reflectivity reflectivity variable name
        var_other : list
            List with other variables that should be masked with the radar
            reflectivity detection limit.
        """

        # apply radar reflectivity detection limit
        ix = self.ds[var_ze] > db2li(self.beam.spec.detection_limit)

        for var in var_other:
            self.ds[var] = self.ds[var].where(ix)

    @staticmethod
    def add_noise(x, x_std, noise):
        """
        Equation to calculate the noise from values without noise, the
        uncertainty of the values, and random noise.

        Parameters
        ----------
        x : xr.DataArray
            Radar reflectivity [dB] or doppler velocity [m s-1]
        x_std : float
            Radar reflectivity uncertainty [dB] or doppler velocity uncertainty
            [m s-1]
        noise : np.array
            Random noise with shape equal to x.

        Returns
        -------
        x_noise : xr.DataArray
            Radar reflectivity with added noise [dB]
        """

        x_noise = x + x_std * noise

        return x_noise

    def calculate_vm_std_nubf(self):
        """
        Calculate outstanding error in correcting Mean Doppler Velocity biases
        caused by non-uniform beam filling

        The calculation is based on the horizontal radar reflectivity gradient
        at the input resolution. The gradient is calculated along the along-
        track direction. The gradient is then averaged onto the satellite grid
        and the absolute value is taken. The error is then calculated as 0.15
        times the gradient divided by 3 dBZ/km. Bins without reflectivity are
        set to 0 before averaging onto satellite resolution.
        """

        # calculate bin edges of satellite grid
        along_track_sat_bin_edges = self.calculate_along_track_sat_bin_edges()
        height_sat_bin_edges = self.calculate_height_sat_bin_edges()

        # calculate horizontal ze gradient on input grid in dBZ/km
        ze_gradient = li2db(self.ds["ze"]).diff("along_track") / (
            self.ds["along_track"].diff("along_track").mean() / 1000
        )

        # fill nan values with zero
        ze_gradient = ze_gradient.fillna(0)
        ze_gradient = (
            ze_gradient.groupby_bins(
                "along_track",
                bins=along_track_sat_bin_edges,
                labels=self.ds["along_track_sat"].values,
            )
            .mean()
            .groupby_bins(
                "height",
                bins=height_sat_bin_edges,
                labels=self.ds["height_sat"].values,
            )
            .mean()
        )
        ze_gradient = ze_gradient.rename(
            {
                "along_track_bins": "along_track_sat",
                "height_bins": "height_sat",
            }
        )

        # calculate absolute value of ze gradient
        ze_gradient = np.abs(ze_gradient)

        vm_std_nubf = 0.15 * ze_gradient / 3

        return vm_std_nubf

    def vm_uncertainty_equation(self, vm_std_broad, vm_std_nubf):
        """
        Calculates the total Doppler velocity uncertainty based on the
        broadening Doppler velocity uncertainty and the non-uniform beam
        filling Doppler velocity uncertainty.

        Based on Equation (4) in

        Parameters
        ----------
        vm_std_broad : float, np.array
            Doppler velocity uncertainty due to broadening [m s-1]
        vm_std_nubf : float, np.array
            Doppler velocity uncertainty due to non-uniform beam filling
            [m s-1]
        """

        vm_std = np.sqrt(vm_std_broad**2 + vm_std_nubf**2)

        return vm_std

    def calculate_ze_noise(self):
        """
        Adds noise to satellite radar reflectivity based on the pre-defined
        lookup table with noise values for different radar reflectivity bins.
        Empty bins are filled with noise according to the noise level.
        """

        # generate noise
        lower = -4.5
        upper = 4.5
        mu = 0
        sigma = 2
        n = np.prod(self.ds["ze_sat"].shape)
        noise = np.array(
            stats.truncnorm.rvs(
                a=(lower - mu) / sigma,
                b=(upper - mu) / sigma,
                loc=mu,
                scale=sigma,
                size=n,
            )
        ).reshape(self.ds["ze_sat"].shape)

        # interpolates discrete standard deviations
        f = interp1d(
            self.beam.spec.ze_bins,
            self.beam.spec.ze_std,
            kind="linear",
            fill_value="extrapolate",  # type: ignore
        )

        # apply noise
        self.ds["ze_sat_noise"] = db2li(
            self.add_noise(
                x=li2db(self.ds["ze_sat"]),
                x_std=f(li2db(self.ds["ze_sat"])),
                noise=noise,
            )
        )

    def calculate_vm_noise(self):
        """
        Adds noise to satellite Doppler velocity based on the pre-defined
        lookup table with noise values for different radar reflectivity bins.

        Note:
        The noise is added to the satellite Doppler velocity with the satellite
        motion error.
        """

        lower = -self.beam.spec.nyquist_velocity
        upper = self.beam.spec.nyquist_velocity
        mu = 0
        sigma = 1
        n = np.prod(self.ds["vm_sat_vel"].shape)
        noise = np.array(
            stats.truncnorm.rvs(
                a=(lower - mu) / sigma,
                b=(upper - mu) / sigma,
                loc=mu,
                scale=sigma,
                size=n,
            )
        ).reshape(self.ds["vm_sat_vel"].shape)

        # interpolates discrete standard deviations
        f = interp1d(
            self.beam.spec.vm_bins_broad,
            self.beam.spec.vm_std_broad,
            kind="linear",
            fill_value="extrapolate",  # type: ignore
        )

        # calculate uncertainty due to broadening
        vm_std_broad = f(li2db(self.ds["ze_sat"]))

        # calculate uncertainty due to non-uniform beam filling
        vm_std_nubf = self.calculate_vm_std_nubf()

        # calculate total Doppler velocity uncertainty
        vm_std = self.vm_uncertainty_equation(
            vm_std_broad=vm_std_broad,
            vm_std_nubf=vm_std_nubf,
        )

        # add Doppler velocity error
        self.ds["vm_sat_noise"] = self.add_noise(
            x=self.ds["vm_sat_vel"], x_std=vm_std, noise=noise
        )

    def fold_vm(self):
        """
        Doppler velocity folding correction.
        """

        # keys: nyquist velocity offset added for folding
        # values: velocity bin edges as multiple of the nyquist velocity
        folding_dct = {
            -2: [1, 3],
            -4: [3, 5],
            -6: [5, 7],
            -8: [7, 9],
            2: [-3, -1],
            4: [-5, -3],
            6: [-7, -5],
            8: [-9, -7],
        }

        # data array with folded velocity
        self.ds["vm_sat_folded"] = self.ds["vm_sat_noise"].copy()

        # folding flag
        self.ds["folding_flag"] = xr.zeros_like(self.ds["vm_sat_noise"])

        for offset, (v0, v1) in folding_dct.items():
            # convert factors to doppler velocity
            v0 = v0 * self.beam.spec.nyquist_velocity
            v1 = v1 * self.beam.spec.nyquist_velocity
            vm_offset = offset * self.beam.spec.nyquist_velocity

            # this is true if folding is applied
            in_interval = (self.ds["vm_sat_folded"] >= v0) & (
                self.ds["vm_sat_folded"] < v1
            )

            # assign folding factor to flag
            self.ds["folding_flag"] = xr.where(
                in_interval,
                1,
                self.ds["folding_flag"],
            )

            # fold velocity within the given interval
            self.ds["vm_sat_folded"] = xr.where(
                in_interval,
                self.ds["vm_sat_folded"] + vm_offset,
                self.ds["vm_sat_folded"],
            )

        # ensure that doppler velocity is within the nyquist velocity
        assert (
            self.ds["vm_sat_folded"].min() >= -self.beam.spec.nyquist_velocity
        ), (
            f"Velocity values below the nyquist velocity: "
            f'{self.ds["vm_sat_folded"].min()}'
        )

        assert (
            self.ds["vm_sat_folded"].max() <= self.beam.spec.nyquist_velocity
        ), (
            f"Velocity values above the nyquist velocity: "
            f'{self.ds["vm_sat_folded"].max()}'
        )

    def add_attributes(self):
        """
        Adds attributes to the variables of the dataset
        """

        # overwrite attributes of ze and vm inputs
        self.ds["ze"].attrs = dict(
            standard_name="radar_reflectivity_factor",
            long_name="Radar reflectivity factor of input",
            units="mm6 m-3",
            description="Radar reflectivity factor of input",
        )

        self.ds["vm"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Mean Doppler velocity of input",
            units="m s-1",
            description="Mean Doppler velocity of input",
        )

        # add attributes to dimensions
        self.ds["along_track"].attrs = dict(
            standard_name="along_track",
            long_name="Along-track",
            units="m",
            description="Along-track distance",
        )

        self.ds["height"].attrs = dict(
            standard_name="height",
            long_name="height",
            units="m",
            description="Height of bin in meters above mean sea level",
        )

        self.ds["along_track_sat"].attrs = dict(
            standard_name="along_track",
            long_name="Along-track",
            units="m",
            description="Along-track distance at satellite resolution",
        )

        self.ds["height_sat"].attrs = dict(
            standard_name="height",
            long_name="height",
            units="m",
            description="Height of bin in meters above mean sea level at "
            "satellite resolution",
        )

        # add attributes to variables
        self.ds["nubf"].attrs = dict(
            standard_name="non_uniform_beam_filling",
            long_name="Non-uniform beam filling",
            units="dBZ",
            description="Non-uniform beam filling calculated as the standard "
            "deviation of radar reflectivity in linear units of the input "
            "data.",
        )

        self.ds["nubf_flag"].attrs = dict(
            standard_name="non_uniform_beam_filling_flag",
            long_name="Non-uniform beam filling flag",
            description="Non-uniform beam filling flag. 1 means non-uniform "
            "beam filling is higher than 1 dB, 0 means non-uniform beam "
            "filling is lower than 1 dB.",
        )

        self.ds["signal_fraction"].attrs = dict(
            standard_name="signal_fraction",
            long_name="Fraction of bins that contain signal",
            description="Fraction of bins that contain signal. 1 means all "
            "bins contain signal, 0 means no bins contain signal.",
        )

        self.ds["ms_flag"].attrs = dict(
            standard_name="multiple_scattering",
            long_name="Multiple scattering flag",
            description="Multiple scattering flag. 1 means multiple "
            "scattering occurs, 0 means no multiple scattering occurs. "
            "This flag only makes sense for airborne observations. "
            "Groundbased observations likely underestimate the occurrence of "
            "multiple scattering due to rain attenuation.",
        )

        self.ds["folding_flag"].attrs = dict(
            standard_name="folding_flag",
            long_name="Folding flag",
            description="Folding flag. 1 means velocity is folded, 0 means "
            "velocity is not folded.",
        )

        self.ds["ze_acon"].attrs = dict(
            standard_name="radar_reflectivity_factor",
            long_name="Convolved radar reflectivity factor",
            units="mm6 m-3",
            description="Convolved radar reflectivity factor",
        )

        self.ds["vm_acon"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Convolved mean Doppler velocity",
            units="m s-1",
            description="Convolved mean Doppler velocity",
        )

        self.ds["vm_acon_err"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Convolved mean Doppler velocity with satellite motion "
            "error",
            units="m s-1",
            description="Convolved mean Doppler velocity with satellite "
            "motion error",
        )

        self.ds["ze_aconint"].attrs = dict(
            standard_name="radar_reflectivity_factor",
            long_name="Convolved and integrated radar reflectivity factor",
            units="mm6 m-3",
            description="Convolved and integrated radar reflectivity factor",
        )

        self.ds["vm_aconint"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Convolved and integrated mean Doppler velocity",
            units="m s-1",
            description="Convolved and integrated mean Doppler velocity",
        )

        self.ds["vm_aconint_err"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Convolved and integrated mean Doppler velocity with "
            "satellite motion error",
            units="m s-1",
            description="Convolved and integrated mean Doppler velocity with "
            "satellite motion error",
        )

        self.ds["ze_sat"].attrs = dict(
            standard_name="radar_reflectivity_factor",
            long_name="Convolved and integrated radar reflectivity factor",
            units="mm6 m-3",
            description="Convolved and integrated radar reflectivity factor"
            "along height and along track",
        )

        self.ds["vm_sat"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Convolved and integrated mean Doppler velocity",
            units="m s-1",
            description="Convolved and integrated mean Doppler velocity"
            "along height and along track",
        )

        self.ds["vm_sat_vel"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Convolved and integrated mean Doppler velocity with "
            "satellite motion error",
            units="m s-1",
            description="Convolved and integrated mean Doppler velocity with "
            "satellite motion error along height and along track",
        )

        self.ds["vm_bias"].attrs = dict(
            standard_name="mean_doppler_velocity_bias",
            long_name="Doppler velocity bias",
            units="m s-1",
            description="Doppler velocity bias between the estimate with and "
            "without satellite motion error. Higher biases occur under higher "
            "non-uniform beam filling.",
        )

        self.ds["ze_sat_noise"].attrs = dict(
            standard_name="radar_reflectivity_factor",
            long_name="Convolved and integrated radar reflectivity factor "
            "with noise",
            units="mm6 m-3",
            description="Convolved and integrated radar reflectivity factor "
            "with noise",
        )

        self.ds["vm_sat_noise"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Convolved and integrated mean Doppler velocity with "
            "noise and satellite motion error",
            units="m s-1",
            description="Convolved and integrated mean Doppler velocity with "
            "noise and satellite motion error",
        )

        self.ds["vm_sat_folded"].attrs = dict(
            standard_name="mean_doppler_velocity",
            long_name="Doppler velocity with noise, satellite motion error, "
            "and folding",
            units="m s-1",
            description="Doppler velocity with noise, satellite motion "
            "error, and folding",
        )

        # time encoding
        self.ds["time"].encoding = dict(
            units="seconds since 1970-01-01 00:00:00",
            calendar="gregorian",
        )
        self.ds["time"].attrs = dict(
            standard_name="time",
            long_name="Time",
        )

        # add variables about satellite
        self.ds["sat_ifov"] = xr.DataArray(
            self.beam.ifov,
            attrs=dict(
                standard_name="sat_ifov",
                long_name="Satellite instantaneous field of view",
                units="m",
                description="Satellite instantaneous field of view",
            ),
        )

        self.ds["sat_range_resolution"] = xr.DataArray(
            self.beam.spec.range_resolution,
            attrs=dict(
                standard_name="sat_range_resolution",
                long_name="Satellite range resolution",
                units="m",
                description="Satellite range resolution",
            ),
        )

        self.ds["sat_along_track_resolution"] = xr.DataArray(
            self.beam.spec.along_track_resolution,
            attrs=dict(
                standard_name="sat_along_track_resolution",
                long_name="Satellite along-track resolution",
                units="m",
                description="Satellite along-track resolution",
            ),
        )

        # global attributes
        self.ds.attrs["title"] = (
            f"{self.beam.spec.name} simulated from "
            f"suborbital observations with orbital-radar {__version__}"
        )
        self.ds.attrs["created"] = str(np.datetime64("now"))
        self.ds.attrs["description"] = (
            "Simulated spaceborne radar reflectivity and Doppler velocity "
            "from suborbital radar data. The forward simulation "
            "follows Kollias et al. (2014) and Lamer et al. (2020)"
        )

    def plot(self, **kwds):
        """
        Along-track plot of the simulated radar reflectivity and Doppler
        velocity.
        """

        fig = plot_along_track(ds=self.ds, **kwds)

        return fig

    def plot_histogram(self, **kwds):
        """
        Histogram plot of the simulated radar reflectivity and Doppler
        velocity.
        """

        fig = plot_histogram(ds=self.ds, **kwds)

        return fig

    def plot_scatter(self, **kwds):
        """
        Scatter plot between satellite data and suborbital data.
        """

        fig = plot_scatter(ds=self.ds, **kwds)

        return fig

    def transform(self, ds):
        """
        Runs the entire simulator.

        Parameters
        ----------
        ds : xarray.Dataset
            Data from suborbital radar interpolated to "along_track" [m] and
            "height" [m] coordinates. The dataset must contain the following
            variables:
            Radar reflectivity "ze" [mm6 m-3],
            Doppler velocity "vm" [m s-1].
            Both variables should have no nan values. Any nan's should be
            filled with zeros.
        """

        # add input dataset to class
        self.ds = ds

        # check input dataset for consistency
        print("Check input dataset")
        self.check_input_dataset()

        # prepare input dataset for computations
        print("Prepare input dataset")
        self.prepare_input_dataset()

        # compute weighting functions
        print("Compute weighting functions")
        self.beam.calculate_weighting_functions(
            range_coords=self.ds["height"],
            along_track_coords=self.ds["along_track"],
        )

        # detection limit
        print("Apply detection limit to input data")
        self.apply_detection_limit(var_ze="ze", var_other=["ze", "vm"])

        # transformations to spaceborne radar
        print("Convolve along track")
        self.convolve_along_track()

        print("Integrate along track")
        self.integrate_along_track()

        print("Convolve height")
        self.convolve_height()

        # detection limit
        print("Apply detection limit on satellite view")
        self.apply_detection_limit(
            var_ze="ze_sat", var_other=["ze_sat", "vm_sat", "vm_sat_vel"]
        )

        # noise
        print("Calculate Ze noise")
        self.calculate_ze_noise()

        print("Calculate Vm noise")
        self.calculate_vm_noise()

        # doppler velocity folding
        print("Fold Vm")
        self.fold_vm()

        # non-uniform beam filling
        print("Calculate non-uniform beam filling")
        self.calculate_nubf()

        # non-uniform beam filling flag
        print("Calculate non-uniform beam filling flag")
        self.calculate_nubf_flag()

        # doppler velocity bias
        print("Calculate Doppler velocity bias")
        self.calculate_vm_bias()

        # doppler velocity bias flag
        print("Calculate Doppler velocity bias flag")
        self.calculate_vm_bias_flag()

        # multiple scattering flag
        print("Calculate multiple scattering flag")
        self.calculate_ms_flag()

        # signal fraction
        print("Calculate signal fraction")
        self.calculate_signal_fraction()

        # set attributes
        print("Add attributes")
        self.add_attributes()
