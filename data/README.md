This folder is aimed to contains all different types of data used for overlap-based MCL

Please use the recommended data structure as follows:

```bash
data
    ├── 07
    │   ├── calib.txt
    │   ├── poses.txt
    │   ├── velodyne
    │   │   ├── 000000.bin
    │   │   ├── 000001.bin
    │   │   └── ...
    │   ├── ground_truth
    │   │   ├── ground_truth_overlap_yaw.npz
    │   │   ├── test_set.npz
    │   │   └── train_set.npz
    │   ├── map
    │   │   ├── depth
    │   │   │   ├── x1_y1.npy
    │   │   │   ├── x1_y2.npy
    │   │   │   └── ...
    │   │   ├── normal
    │   │   │   ├── x1_y1.npy
    │   │   │   ├── x1_y2.npy
    │   │   │   └── ...
    │   │   └── feature_volumes
    │   │       ├── x1_y1.npz
    │   │       ├── x1_y2.npz
    │   │       └── ...
    │   ├── query
    │   │   ├── depth
    │   │   │   ├── 000000.npy
    │   │   │   ├── 000001.npy
    │   │   │   └── ...
    │   │   ├── normal
    │   │   │   ├── 000000.npy
    │   │   │   ├── 000001.npy
    │   │   │   └── ...
    │   │   └── feature_volumes
    │   │       ├── 000000.npz
    │   │       ├── 000001.npz
    │   │       └── ...
    │   └── training
    │       ├── depth
    │       │   ├── 000000.npy
    │       │   ├── 000001.npy
    │       │   └── ...
    │       └── normal
    │           ├── 000000.npy
    │           ├── 000001.npy
    │           └── ...
    └── model.weight
```

