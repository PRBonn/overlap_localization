#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates depth and normal data for map grids

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm


try:
  from c_gen_depth_and_normal import gen_depth_and_normal
except:
  print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
  sys.exit(-1)


def gen_depth_and_normal_map(virtual_scan_folder, depth_folder, normal_folder, range_image_params):
  """ Generate depth and normal data given virtual grid scans.
    Args:
      virtual_scan_folder: path of virtual scan folder.
      depth_folder: path of folder for generated depth data.
      normal_folder: path of folder for generated normal data.
      range_image_params: parameters for generating a range image.
  """
  # load virtual scans
  virtual_scan_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(virtual_scan_folder)) for f in fn]
  virtual_scan_paths.sort()
  virtual_scan_paths = np.array(virtual_scan_paths)
  
  if not os.path.exists(depth_folder):
    os.makedirs(depth_folder)
    
  if not os.path.exists(normal_folder):
    os.makedirs(normal_folder)
  
  print('start generating depth and normal data for map scans...')
  for virtual_scan_path in tqdm(virtual_scan_paths):
    # load virtual scan and the coordinate
    virtual_scan = np.load(virtual_scan_path)['arr_0']
    coordinate = os.path.basename(virtual_scan_path).replace('.npz', '')

    # check existence
    if os.path.exists(os.path.join(depth_folder, coordinate + '.npy')):
      print('existing: ', coordinate)
      continue
    
    # generate depth and normal data
    depth_and_normal = gen_depth_and_normal(virtual_scan.astype(np.float32),
                                            range_image_params['height'], range_image_params['width'],
                                            range_image_params['fov_up'], range_image_params['fov_down'],
                                            range_image_params['max_range'], range_image_params['min_range'])
    
    depth = depth_and_normal[:, :, 3] / np.max(depth_and_normal[:, :, 3])
    normal = depth_and_normal[:, :, :3]
    
    # save depth and normal data
    np.save(os.path.join(depth_folder, coordinate), depth)
    np.save(os.path.join(normal_folder, coordinate), normal)


if __name__ == '__main__':
  # load config file
  config_filename = '../../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  virtual_scan_folder = '../' + config['virtual_scan_folder']
  depth_folder = '../' + config['map_depth_folder']
  normal_folder = '../' + config['map_normal_folder']
  
  range_image_params = config['range_image']

  gen_depth_and_normal_map(virtual_scan_folder, depth_folder, normal_folder, range_image_params)


