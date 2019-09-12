#!/bin/sh

python ../src/scuba_grid.py example_scuba_grid.yaml -v info
python ../src/scuba_alongtrack.py example_scuba_alongtrack.yaml -v info
python ../src/scuba_mooring.py example_scuba_mooring.yaml -v info
python ../src/scuba_tide_gauge.py example_scuba_tide_gauge.yaml -v info
python ../src/scuba_statistic.py example_scuba_stats.yaml -v info
