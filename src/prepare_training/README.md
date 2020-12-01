# Procedure for generating training data for overlap localization

## Installation

### C++ library for generating training data

We implemented the generation of depth and normal maps in C++. In order to call it from python, we are
using the [pybind11 library](https://github.com/pybind/pybind11). At least version 2.2 is required.

We recommended to use pip to install the library, e.g.

```
sudo -H pip3 install pybind11
```
(The package python3-pybind11 from the Ubuntu repositories maybe too old).

Our C++ code can be build with

```
cd src/prepare_training/c_utils
mkdir build && cd build
cmake ..
make
```

Note that depending on the setup of the pybind11 library, one has to give the path to the `.cmake` files
for the pybind library, e.g.:

```
cmake .. -Dpybind11_DIR=/usr/local/lib/python3.6/dist-packages/pybind11/share/cmake/pybind11
```
Or, one could add pybind11 as a subdirectory inside the c++ project and directly compile it. 
For more details we refer to the pybind11 compiling [doc](https://pybind11.readthedocs.io/en/stable/compiling.html).

To use the C++ library, one needs to specify the path of the library by:

```
export PYTHONPATH=$PYTHONPATH:<path-to-library>
``` 

## Usage

### generate training data

For a quick training demo, one could download the training data ([download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/training.zip)) of KITTI sequence 07 preprocessed by us and directly train a model by running:

```bash
python3 ../OverlapNet/src/two_heads/training.py ../config/localization.yml
```

Here we also give an example to generate training data for using OverlapNet to train a sensor model from scratch (will take a longer time). 

1. Download the KITTI dataset sequence 07, [download](http://www.ipb.uni-bonn.de/html/projects/overlap_mcl/kitti_07.zip).
2. Run `python3 main_prepare_training.py` to generate the data step by step.
3. Adapt the OverlapNet configuration file. Use `07` as sequence name and set the correct folder for the data root folder. The recommended data structure can be found in data structure [README.md](../../data/README.md)
4. Train the model following the steps mentioned in [OverlapNet](https://github.com/PRBonn/OverlapNet).


