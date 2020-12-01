#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script reindexes the grid virtual frames for training.

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import shutil


def convert_training_data(config):
  """ Convert the training data into OverlapNet format.
    Args:
      config: configuration parameters.
  """
  raw_depth_folder = config['map_depth_folder']
  raw_normal_folder = config['map_normal_folder']
  overlap_yaw_file = config['ground_truth']
  lut_path = config['rename_lut']
  grid_resolution = config['resolution']
  
  new_depth_folder = config['training_depth_folder']
  if not os.path.exists(new_depth_folder):
    os.makedirs(new_depth_folder)
    
  new_normal_folder = config['training_normal_folder']
  if not os.path.exists(new_normal_folder):
    os.makedirs(new_normal_folder)
  
  # load overlap and yaw labels
  overlap_yaw = np.load(overlap_yaw_file)['arr_0']  # [current, reference, overlap, yaw, x, y]

  # load the renaming look up table
  rename_lut = np.load(lut_path)['arr_0']
  
  # load all paths of virtual frames
  depth_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(raw_depth_folder)) for f in fn]
  depth_paths.sort()
  depth_paths = np.array(depth_paths)
  
  normal_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(raw_normal_folder)) for f in fn]
  normal_paths.sort()
  normal_paths = np.array(normal_paths)
  
  # collect grid coords
  grid_coords = []
  for depth_path in depth_paths:
    grid_coords.append(os.path.basename(depth_path).replace('.npy', '').split('_'))
  grid_coords = np.array(grid_coords, dtype=float)
  
  grid_coords = np.round(grid_coords / grid_resolution)
  
  grid_x_coord_max = round(np.max(overlap_yaw[:, 4]) / grid_resolution)
  grid_x_coord_min = round(np.min(overlap_yaw[:, 4]) / grid_resolution)
  grid_y_coord_max = round(np.max(overlap_yaw[:, 5]) / grid_resolution)
  grid_y_coord_min = round(np.min(overlap_yaw[:, 5]) / grid_resolution)
  
  # only copy needed frames
  arg_new_index = np.argwhere(rename_lut >= 0)
  
  for inverse_coords in tqdm(arg_new_index):
    inverse_x = inverse_coords[1]
    inverse_y = inverse_coords[0]
    x_coord = round(grid_x_coord_min + inverse_x)
    y_coord = round(grid_y_coord_max - inverse_y)
    
    file_idx = np.argwhere((grid_coords[:, 0] == x_coord) & (grid_coords[:, 1] == y_coord))

    old_depth_path = str(np.squeeze(depth_paths[file_idx[0]]))
    old_normal_path = str(np.squeeze(normal_paths[file_idx[0]]))
    
    if rename_lut[inverse_y, inverse_x] >= 0:
      new_idx = rename_lut[inverse_y, inverse_x]
      
      # copy
      shutil.copy(old_depth_path, new_depth_folder)
      shutil.copy(old_normal_path, new_normal_folder)
      
      # rename
      old_depth_name = os.path.join(new_depth_folder, os.path.basename(old_depth_path))
      old_normal_name = os.path.join(new_normal_folder, os.path.basename(old_normal_path))
      new_depth_name = os.path.join(new_depth_folder, str(new_idx).zfill(6) + '.npy')
      new_normal_name = os.path.join(new_normal_folder, str(new_idx).zfill(6) + '.npy')
      os.rename(old_depth_name, new_depth_name)
      os.rename(old_normal_name, new_normal_name)


if __name__ == '__main__':
  # load config file
  config_filename = '../../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
   
  convert_training_data(config)
