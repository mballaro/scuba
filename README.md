
![SCUBA Logo](https://github.com/mballaro/scuba/blob/master/share/scuba_files/logo.png)

SCUBA performs spectral analysis of along-track and gridded dataset, as well as spectral statistical comparison between two fields (e.g., along-track vs grid, grid vs grid).

# Table of contents
===================
<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Structure of SCUBA](#structure-of-scuba)
   * [Usage & Background ](#usage)
   * [First step with SCUBA](#first-step-with-scuba)
      * [Running test cases](#running-test-cases)
      * [Visualisation of the results](#visualisation-of-the-results)
   * [Authors & Contributors](#authors-contributors)
<!--te-->


# Structure of SCUBA
====================
```
                                        SCUBA
                                          |
            +--------------------+----------------+---------------------+
          share                 src           test_case                tools
```
* `share` contains information or files for running the program, e.g. the distance from closest land point (needed for coastal editing) or altimeter mission information
* `src` contains the python scripts to perform the analysis
* `test_case` provides test cases to test the program
* `tools` includes scripts to display spectrum, resolution and spectral Taylor Diagram 


# Usage & Background
====================

* `scuba_alongtrack.py` performs spectral analysis on along-track data or between along-track and gridded data
* `scuba_grid.py` performs spectral analysis on gridded data or between two gridded data

The program is structured as follow:
* 1- reading the dataset

* 2- computing segment (along-track, or zonal or meridional) database

<p align="center">
<b>Example alongtrack direction</b>
</p>

![segment along track](https://github.com/mballaro/scuba/blob/master/share/scuba_files/example_segment_alongtrack_direction.gif)

<p align="center">
<b>Example zonal direction</b>
</p>

![segment zonal](https://github.com/mballaro/scuba/blob/master/share/scuba_files/example_segment_x_direction.gif)

<p align="center">
<b>Example meridional direction</b>
</p>

![segment meridional](https://github.com/mballaro/scuba/blob/master/share/scuba_files/example_segment_y_direction.gif)

* 3- performing spectral analysis in boxes by selecting all the segments of the database found within the box
![segment selection](https://github.com/mballaro/scuba/blob/master/share/scuba_files/example_selection.gif)


* 4- gridding the results

* 5- writing netCDF output


# First step with SCUBA
=======================
## Running test cases
---
     >> cd test_case/
     >> python ../src/scuba_grid.py example_scuba_grid.yaml
	 >> python ../src/scuba_alongtrack.py example_scuba_alongtrack.yaml

This test cases performs spectral analysis on altimeter maps and along-track data. For more detail on the analysis parameters see *.yaml parameter files.

## Visualisation of the results
---
     >> cd tools
     >> python display_resolution.py ../test_case/psd_alongtrack_direction.nc

![RESOLUTION](https://github.com/mballaro/scuba/blob/master/share/scuba_files/resolution.png)

     >> python display_scuba_alongtrack.py ../test_case/psd_alongtrack_direction.nc
     
![SPECTRAL](https://github.com/mballaro/scuba/blob/master/share/scuba_files/spectrum.png)

     >> python display_spectral_taylor_diagram.py ../test_case/psd_alongtrack_direction.nc

![TD](https://github.com/mballaro/scuba/blob/master/share/scuba_files/spectral_taylor_diagram.png)


# Authors & Contributors
========================
* Maxime Ballarotta, Clément Ubelmann
* _Feel free to dive in ..._

