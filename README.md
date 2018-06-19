
![SCUBA Logo](https://github.com/mballaro/scuba/blob/master/share/scuba_files/logo.png)

# Structure of SCUBA (SpeCtral Utility Belt for Altimetry application)
```
                                        SCUBA
                                          |
            +--------------------+----------------+---------------------+
	  share		        src	      test_case	              tools
```
* `share` contains information or files for running the program, e.g. the distance from closest land point (needed for coastal editing) or altimeter mission information
* `src` contains the python script to perform the analysis
* `test_case` provides a test_case to test the program
* `tools` includes scripts to display spectrum, resolution and spectral Taylor Diagram 

# First step with SCUBA
A test case:
--
     >> cd test_case/
     >> python ../src/scuba.py template.yaml

This test case will perform a spectral analysis on altimeter maps and along-track data. For more detail on the analysis parameter see template.yaml.

# Content of the netCDF output
```
netcdf test_case_al {
dimensions:
	f = 106 ;
	y = 180 ;	
	x = 360 ;
variables:
	double frequency(f, y, x) ;
	double spectrum_along_track(f, y, x) ;
	double spectrum_map(f, y, x) ;
	double psd_along_track(f, y, x) ;
	double psd_map(f, y, x) ;
	double spectrum_diff_at_map(f, y, x) ;
	double psd_diff_at_map(f, y, x) ;
	double coherence(f, y, x) ;
	double effective_resolution(y, x) ;
	double useful_resolution(y, x) ;
	double nb_segment(y, x) ;
	double lat2D(y, x) ;
	double lon2D(y, x) ;
	double lat(y) ;
	double lon(x) ;
}
```
x: longitude dimension
y: latitude dimension
f: wavelength dimension
frequency: wavelenght at any lon/lat point
spectrum_along_track: spectrum of along-track computed at any lon/lat point
spectrum_map: spectrum of map computed at any lon/lat point
psd_along_track: power spectrum density of along-track computed at any lon/lat point
psd_map: power spectrum density of map computed at any lon/lat point
spectrum_diff_at_map: spectrum of difference along-track - map computed at any lon/lat point
psd_diff_at_map: power spectrum density of difference along-track - map computed at any lon/lat point
coherence: magnitude squared coherence at any lon/lat point
effective_resolution: effective_resolution at any lon/lat point
useful_resolution: useful resolution at any lon/lat point
nb_segment: segment number used in the spectral computations
lat2D: output grid lat 2D
lon2D: output grid lon 2D
lat: output grid lat 1D
lon:output grid lon 1D


# Visualisation of the results
     >> cd tools
     >> python display_resolution.py ../test_case/test_case.nc

![RESOLUTION](https://github.com/mballaro/scuba/blob/master/share/scuba_files/resolution.png)

     >> python display_spectrum_coherence.py ../test_case/test_case.nc
     
![SPECTRAL](https://github.com/mballaro/scuba/blob/master/share/scuba_files/spectrum.png)

     >> python display_spectral_taylor_diagram.py ../test_case/test_case.nc

![TD](https://github.com/mballaro/scuba/blob/master/share/scuba_files/spectral_taylor_diagram.png)
