Getting Started with Orbital-Radar
===================================

After successfully installing the software, follow these steps to acquire the 
test data and create the configuration file. You can then run the 
simulator either from the command line or another Python script. An example 
Python script is available under `examples/groundbased.ipynb` for the test 
dataset.

Download Test Data
------------------

The test data includes:

- One day of groundbased radar and cloudnet data from Mindelo, Cape Verde, 6 April 2022.
- One day of groundbased radar and cloudnet data for JOYCE, Jülich, Germany, 6 April 2021.
- One flight track from MIRAC-A on board of Polar-5 Aircraft, AFLUX campaign, 1 April 2019 
  (Mech et al. 2022, https://doi.org/10.1038/s41597-022-01900-7)
- Range weighting function of the EarthCARE CPR.

You can download it from sciebo at the following link: 
`Download Test Data <https://uni-koeln.sciebo.de/s/amrLECxo1Ifretu>`__.

After downloading, please unzip the 'orbital_radar_data.zip' archive and 
store it a location of your choice. Lateron we will define the path to the 
test data in the configuration file.

Create Configuration File
-------------------------

The configuration file contains all the necessary settings for the Orbital 
Radar software. A template of the configuration file is provided on GitHub 
for reference. Follow these steps:

#. Copy the configuration file example ``orbital_radar_config.toml.example`` 
   from the root directory of this package to a location of your choice.
#. Rename the copied example file to ``orbital_radar_config.toml``.
#. Modify the following keys to get started:

   * ``radar.file_earthcare``: Set the absolute path to the EarthCARE range 
     weighting function. The range weighting function is provided in the test 
     data archive. If you keep the test data structure unchanged, the filename
     ends with
     ``orbital_radar_data/range_weighting_function/CPR_PointTargetResponse.txt``

   * ``paths.min.radar``: Set the path for the Mindelo radar data. 
     The radar data are provided in the test data archive. 
     If you keep the test data structure unchanged, the path ends with
     ``orbital_radar_data/groundbased/min/radar``. 
     The date structure in the radar folder should remain unchanged.

   * ``paths.min.cloudnet``: Set the path for the Mindelo cloudnet data.
     The cloudnet data are provided in the test data archive. 
     If you keep the test data structure unchanged, the path ends with
     ``orbital_radar_data/groundbased/min/cloudnet``.
     The date structure in the cloudnet folder should remain unchanged.

   * ``paths.min.output``: Set the path for the simulation output for the 
     Mindelo station. The output NetCDF file will be stored in this directory.


Set Environment Variable
------------------------

To streamline the configuration process, it is recommended to set the 
environment variable ``ORBITAL_RADAR_CONFIG_PATH``, which specifies the 
directory to the configuration file. Follow these steps:

1. Rename the ``.env.example`` file in the source directory to ``.env``.
2. Add the desired path into the ``.env`` file.

Alternatively, you can set the environment variable manually:

.. code-block:: bash

    export ORBITAL_RADAR_CONFIG_PATH=/path/to/config/file

You can create multiple configuration files lateron and keep them inside the
same directory. The environment variable will then allow you to switch between
them easily.


Running the Simulator
---------------------

Orbital-radar can be executed from a Python script, allowing flexible 
use of the simulation and visualization tools. The following example 
demonstrates how to simulate the test data:

.. code-block:: python

    from orbital_radar import GroundBased

    # Initialize the groundbased simulator
    grb = GroundBased(
        name="min",
        config_file="orbital_radar_config.toml",
    )

    # Run the groundbased simulator
    grb.run_date(date="2022-02-14", write_output=True)

The output of the simulation will be written to a NetCDF file to the path
specified in the config file.

The output file name is generated automatically:
``ora_0.0.2_earthcare_l1_groundbased_min_inoe94_20220214T000000_20220214T235959.nc``

It contains the following information:
- ``ora``: Orbital Radar
- ``0.0.2``: Version number
- ``earthcare``: Satellite name
- ``l1``: Level of the data
- ``groundbased``: Type of the input data (groundbased or airborne)
- ``min``: Abbrevated name of the groundbased radar site
- ``inoe94``: Abbrevated name of the radar
- ``20220214T000000``: Start time of the input data
- ``20220214T235959``: End time of the input data

A more comprehensive example is available under `examples/groundbased.ipynb`.


Command Line Interface
----------------------

Orbital-radar comes with a command line interface upon installation. You can 
check out the available options using the following command:

.. code-block:: bash

    orbital-radar --help

This command simulates the groundbased test dataset using the following syntax:

.. code-block:: bash

    orbital-radar -g groundbased -n min -d 2022-02-14 -f orbital_radar_config.toml

.. warning::
    
    If the environment variable ``ORBITAL_RADAR_CONFIG_PATH`` is set via a 
    ``.env`` file, it might not be recognized by the command line interface.
    In this case, please specify the absolute path to the configuration file 
    using the ``-f`` option or export the environment variable manually.


Configuration File Description
------------------------------

The following table describes the configuration file in detail.

.. list-table:: Configuration file description
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Description
   * - ``sat_name``
     - Name of satellite in lowercase (e.g. earthcare)
   * - ``file_earthcare``
     - File path for EarthCARE range weighting function
   * - ``ms_threshold``
     - Multiple-scattering threshold before calculating integral
   * - ``ms_threshold_integral``
     - Multiple-scattering threshold for TOA to surface integral
   * - ``k2``
     - Dielectric constant for calculation of satellite radar reflectivity
   * - ``prepare.attenuation_correction``
     - Use cloudnet for attenuation correction (true or false)
   * - ``prepare.cloudnet_product``
     - Cloudnet product (categorize or ecmwf)
   * - ``prepare.range_min``
     - Minimum range for homogenizing suborbital radar
   * - ``prepare.range_max``
     - Maximum range for homogenizing suborbital radar
   * - ``prepare.range_res``
     - Range resolution for homogenizing suborbital radar
   * - ``prepare.keep_time``
     - Keep time information of suborbital radar (true or false)
   * - ``prepare.ground_echo_ze_max``
     - Maximum reflectivity for ground echo
   * - ``prepare.ground_echo_pulse_length``
     - Ground echo pulse length (for generating artificual ground echo)
   * - ``prepare.groundbased.mean_wind``
     - Mean wind to convert ground-based radar to along-track
   * - ``prepare.airborne.mean_flight_velocity``
     - Mean flight velocity to convert airborne radar to along-track
   * - ``paths.bco.radar``
     - Radar data path for BCO radar
   * - ``paths.bco.cloudnet``
     - Cloudnet data path for BCO radar
   * - ``paths.bco.output``
     - Output data path for BCO radar
   * - ``paths.mp5.radar``
     - Radar data path for MP5 radar
   * - ``paths.mp5.cloudnet``
     - Cloudnet data path for MP5 radar
   * - ``paths.mp5.output``
     - Output data path for MP5 radar
   * - ``attributes.name``
     - Name of the user
   * - ``attributes.affiliation``
     - User's affiliation
   * - ``attributes.address``
     - User's address
   * - ``attributes.mail``
     - User's email address
   * - ``groundbased.bco.name``
     - Name of the BCO ground-based radar
   * - ``groundbased.bco.frequency``
     - Frequency of the BCO radar
   * - ``groundbased.bco.site``
     - Site of the BCO radar
   * - ``groundbased.bco.location``
     - Location of the BCO radar
   * - ``groundbased.bco.k2``
     - K2 parameter for the BCO radar
   * - ``groundbased.bco.lon``
     - Longitude of the BCO radar
   * - ``groundbased.bco.lat``
     - Latitude of the BCO radar
   * - ``groundbased.bco.alt``
     - Altitude of the BCO radar
   * - ``airborne.mp5.name``
     - Name of the Polar 5 radar
   * - ``airborne.mp5.frequency``
     - Frequency of the Polar 5 radar
   * - ``airborne.mp5.platform``
     - Platform of the Polar 5 radar
   * - ``airborne.mp5.k2``
     - K2 parameter for the Polar 5 radar
