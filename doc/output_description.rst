Variable List - Output file
===========================

The orbital-radar package transforms radar observations from ground or aircraft
into the satellite perspective.

.. list-table:: Overview of output variables
   :widths: 25 75
   :header-rows: 1

  * - Variable
    - Description
  * - *sat_ifov*
    - Satellite instantaneous field of view
  * - *sat_range_resolution*
    - Satellite range resolution
  * - *sat_along_track_resolution*
    - Satellite along-track resolution
  * - *mean_wind*
    - Mean horizontal wind
  * - *ze*
    - Radar reflectivity
  * - *vm*
    - Doppler velocity
  * - *ze_sat*
    - Convolved and integrated radar reflectivity factor along range and along track
  * - *vm_sat*
    - Convolved and integrated mean Doppler velocity along range and along track
  * - *vm_sat_vel*
    - Convolved and integrated mean Doppler velocity with satellite motion error along range and along track
  * - *ze_sat_noise*
    - Convolved and integrated radar reflectivity factor with noise
  * - *vm_sat_noise*
    - Convolved and integrated mean Doppler velocity with noise and satellite motion error
  * - *vm_sat_folded*
    - Doppler velocity with noise, satellite motion error, and folding
  * - *nubf_flag*
    - Fraction of bins that contain signal
  * - *ms_flag*
    - Multiple scattering flag
  * - *time*
    - Time
  * - *range*
    - Range distance in meters above mean sea level
  * - *along_track*
    - Along-track distance
  * - *along_track_sat*
    - Along-track distance at satellite resolution
  * - *range_sat*
    - Range distance in meters above mean sea level at satellite resolution

For the forward simulation based on EarthCARE specification: 
Outputted data fields correspond to the L1 EarthCARE products present in C-FMR C-CCD.  

.. list-table:: Overview of Ze-output variables corresponding to C-PRO C-FMR PDD
   :widths: 25 75
   :header-rows: 1

  * - *ze_sat_noise*
    - Z_JSG; reflectivity_no_attenuation_correction; 
      Attenuated reflectivity factor: Observed Radar Reflectivity Factor sampled at JSG
  * - *ms_flag*
    - flagMS; multiple_scatter ing_status
  * - *along_track_sat*
    - along_track; Number of the JSG along-track pixels (in km, 0 km is start of the input-data file)
  * - *range_sat*
    - CPR_height ; Number of vertical levels (bin have negative height)


.. list-table:: Overview of vm-output variables corresponding to C-PRO C-CD PDD
   :widths: 25 75
   :header-rows: 1

  * - *vm_sat_noise*
    - V_D,U; doppler_velocity_best_estimate; 
      Doppler velocity best estimate: Unfolded Doppler velocity integrated corrected for antenna mis-pointing
  * - *vm_sat_folded* 
    - V_d; doppler_velocity_uncorrected; 
      Doppler velocity at JSG after quality control: Doppler velocity after sampling and preliminary quality control
  * - *along_track_sat*
    - along_track; Number of the JSG along-track pixels (in km, 0 km is start of the input-data file)
  * - *range_sat*
    - CPR_height ; Number of vertical levels (bin have negative height)