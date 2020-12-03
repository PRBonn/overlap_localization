# Overlap-based 3D LiDAR Monte Carlo Localization

This directory contains the source code of overlap-based MCL.

### Introduction for each module

The implementation of overlap-based MCL contains three parts:
1. Generating data for training a sensor model using OverlapNet;
2. Generating feature-volume-based map;
3. Monte Carlo localization system using the trained sensor model and feature-volume map.

#### Preparing training data

To train a new observation model, one could find more information in prepare_training [README.md](prepare_training/README.md).

#### Generating map

To generate the feature-volume map, one should first download/train the overlap-based observation [model](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/model.weight), [map data](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/map.zip) and [query data](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/query.zip). The data contains the range depth images and normal images of the map and the query scans.

After putting them all into the `/data` folder, one can get the feature volumes by running:

```bash
python3 gen_feature_volumes.py
```

Please first check the recommended data structure in the data [README.md](../data/README.md) if you get any issues when generating the map.

#### Run overlap-based MCL

Once the map of feature volumes is generated, one could run the overlap-based MCL by:

```bash
python3 main_overlap_mcl.py
```

More technical details could be found in our IROS2020 [paper](http://www.ipb.uni-bonn.de/pdfs/chen2020iros.pdf).

More information about the parameters of the overlap-based observation model and MCL can be found in our configuration file [localization.yml](../config/localization.yml).


