#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the main script file for preparing the map and training for overlap-based MCL.

import sys
import os
import yaml
import numpy as np
import open3d as o3d
import utils

from prepare_training.gen_virtual_scan import gen_pcd_map, rasterize_map
from prepare_training.gen_depth_and_normal_map import gen_depth_and_normal_map
from prepare_training.gen_depth_and_normal_query import gen_depth_and_normal_query
from prepare_training.com_overlap_grid import com_overlaps
from prepare_training.normalize_data import normalize_data
from prepare_training.convert_training_labels import convert_training_labels
from prepare_training.split_train_val import split_train_val
from prepare_training.add_seq_label import add_path_label
from prepare_training.convert_training_data import convert_training_data
  
if __name__ == '__main__':
  # load config file
  config_filename = '../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # specify parameters
  resolution = config['resolution']  # resolution of grid, default is 20 cm
  offset = config['offset']   # for each frame, we generate grids inside a (per default) 1m*1m square
  mcl_only = config['mcl_only']
  range_image_params = config['range_image']
  
  # specify the output paths
  map_file = config['map_file']
  virtual_scan_folder = config['virtual_scan_folder']
  map_depth_folder = config['map_depth_folder']
  map_normal_folder = config['map_normal_folder']
  query_depth_folder = config['query_depth_folder']
  query_normal_folder = config['query_normal_folder']
  overlap_file_path = config['overlap_file_path']
  overlap_yaw_ground_truth_path = config['ground_truth']
  rename_lut_path = config['rename_lut']
  num_frames = config['num_frames']
  
  # load poses
  pose_file = config['pose_file']
  poses = np.array(utils.load_poses(pose_file))
  inv_frame0 = np.linalg.inv(poses[0])
  
  # load calibrations
  calib_file = config['calib_file']
  T_cam_velo = utils.load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)
  
  # convert kitti poses from camera coord to LiDAR coord
  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  poses = np.array(new_poses)
  
  # load LiDAR scans
  scan_folder = config['scan_folder']
  scan_paths = utils.load_files(scan_folder)

  # test for the first N scans
  if num_frames >= len(poses) or num_frames <= 0:
    print('generate training data for all frames with number of: ', len(poses))
  else:
    poses = poses[:num_frames]
    scan_paths = scan_paths[:num_frames]
  
  print("================================================================================")
  print(" step1: build the pcd map ...")
  if os.path.exists(map_file):
    pcd_map = o3d.io.read_point_cloud(map_file)
    num_points = len(pcd_map.points)
    if num_points > 0:
      print('Successfully load pcd map with point size of: ', num_points)
  else:
    print('Creating a pcd map...')
    pcd_map = gen_pcd_map(poses, scan_paths, map_file, vis_map=False)
  
  print(" ")
  print("================================================================================")
  print(" step2: generate virtual scans ...")
  rasterize_map(poses, pcd_map, virtual_scan_folder, resolution, offset, range_image_params)
  
  print(" ")
  print("================================================================================")
  print(" step3: generate depth and normal data for map scans ...")
  gen_depth_and_normal_map(virtual_scan_folder, map_depth_folder, map_normal_folder, range_image_params)
  
  print(" ")
  print("================================================================================")
  print(" step4: generate depth and normal data for query scans ...")
  gen_depth_and_normal_query(scan_paths, query_depth_folder, query_normal_folder, range_image_params)
  
  if not mcl_only:
    print(" ")
    print("================================================================================")
    print(" step5: generate raw overlap ground truth ...")
    com_overlaps(virtual_scan_folder, poses, scan_paths, overlap_file_path, range_image_params)
    
    print(" ")
    print("================================================================================")
    print("  step6: normalize the overlap distribution ...")
    normalize_data(overlap_file_path)
    
    print(" ")
    print("================================================================================")
    print("  step7: convert raw overlap ground truth into OverlapNet format")
    convert_training_labels(overlap_file_path, overlap_yaw_ground_truth_path,
                            rename_lut_path, poses, resolution)
    
    print(" ")
    print("================================================================================")
    print(" step8: convert data into OverlapNet format")
    convert_training_data(config)
    
    print(" ")
    print("================================================================================")
    print(" step9: split into training and validation set ...")
    split_train_val(overlap_yaw_ground_truth_path)
    
    print(" ")
    print("================================================================================")
    print(" step10: add sequence labels into ground truth files ...")
    add_path_label('07/query', '07/training', overlap_yaw_ground_truth_path)
