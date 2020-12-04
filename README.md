# Overlap-based 3D LiDAR Monte Carlo Localization


This repo contains the code for our IROS2020 paper: Learning an Overlap-based Observation Model for 3D LiDAR Localization.
 
It uses the [OverlapNet](https://github.com/PRBonn/OverlapNet) to train an observation model for Monte Carlo Localization and achieves global localization with 3D LiDAR scans.

Developed by [Xieyuanli Chen](http://www.ipb.uni-bonn.de/people/xieyuanli-chen/) and [Thomas Läbe](https://www.ipb.uni-bonn.de/people/thomas-laebe/).


<img src="data/demo.gif" width="800">

Localization results of overlap-based Monte Carlo Localization.

## Publication
If you use our implementation in your academic work, please cite the corresponding [paper](http://www.ipb.uni-bonn.de/pdfs/chen2020iros.pdf):
    
	@inproceedings{chen2020iros,
		author = {X. Chen and T. L\"abe and L. Nardi and J. Behley and C. Stachniss},
		title = {{Learning an Overlap-based Observation Model for 3D LiDAR Localization}},
		booktitle = iros,
		year = {2020},
		url={http://www.ipb.uni-bonn.de/pdfs/chen2020iros.pdf},
	}

## Dependencies

We are using standalone Keras with a TensorFlow backend as a library for neural networks. 

The code was tested with Ubuntu **18.04** with its standard python version **3.6**.

In order to do training and testing on a whole dataset, you need an Nvidia GPU.

To use a GPU, first you need to install the Nvidia driver and CUDA, so have fun!

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- System dependencies:

  ```bash
  sudo apt-get update 
  sudo apt-get install -y python3-pip python3-tk
  sudo -H pip3 install --upgrade pip
  ```

- Python dependencies (may also work with different versions than mentioned in the requirements file)

  ```bash
  sudo -H pip3 install -r requirements.txt
  ```

- OverlapNet: To use this implementation, one needs to clone OverlapNet to the local folder by:
  ```bash
  git clone https://github.com/PRBonn/OverlapNet
  ```

## How to use

#### Quick use
For a quick demo, one could download the [feature volumes](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/feature_volumes.zip) and pre-trained sensor [model](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/model.weight), extract the feature volumes in the `/data` folder following the recommended [data structure](data/README.md), and then run:
  ```bash
  cd src/
  python3 main_overlap_mcl.py
  ```
One could then get the online visualization of overlap-based MCL as shown in the gif.

#### More detailed usage
For more details about the usage and each module of this implementation, one could find them in MCL [README.md](src/README.md).

#### Train a new observation model
To train a new observation model, one could find more information in prepare_training [README.md](src/prepare_training/README.md).

#### Collection of preprocessed data
- KITTI Odometry Sequence 07: [download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/kitti_07.zip).
- Pre-trained Sensor Model: [download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/model.weight).
- Feature Volumes: [download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/feature_volumes.zip).
- Map Data: [download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/map.zip).
- Query Data: [download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/query.zip).
- Training Data: [download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/training.zip).

## License

Copyright 2020, Xieyuanli Chen, Thomas Läbe, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

This project is free software made available under the MIT License. For details see the LICENSE file.


