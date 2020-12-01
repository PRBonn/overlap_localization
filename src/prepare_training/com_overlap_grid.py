#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script computes the ground truth overlap between current frame and grid virtual frames.

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

import utils

try:
  from c_gen_virtual_scan import gen_virtual_scan
except:
  print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
  sys.exit(-1)


def com_overlap(frame_idx, grid_coords, virtual_scan_folder, current_pose,
                current_scan_path, range_image_params, dist_thres=10):
  """ Compute the ground truth overlap values for a given frame with respect to virtual scans.
    Args:
      frame_idx: the index of the given scan.
      grid_coords: coordinates of grids.
      virtual_scan_folder: path of virtual scan folder.
      current_pose: ground truth pose of the given scan.
      current_scan_path: path of the given scan.
      range_image_params: parameters for generating a range image.
      dist_thres: the distance threshold to decide the neighbor virtual scans.
    
    return:
      overlaps: the ground truth overlaps for the given scan with respect to virtual scans.
  """
  # generate current range image
  current_scan = utils.load_vertex(current_scan_path)
  current_vertex = gen_virtual_scan(current_scan.astype(np.float32),
                                    range_image_params['height'], range_image_params['width'],
                                    range_image_params['fov_up'], range_image_params['fov_down'],
                                    range_image_params['max_range'], range_image_params['min_range'])
  current_range = current_vertex[:, :, 3]
  
  valid_num = len(current_range[(current_range > range_image_params['min_range']) &
                                (current_range <= range_image_params['max_range'])])
  
  # select grids that used to calculate the overlap for the current frame
  relative_coords = grid_coords - current_pose[:2, 3]
  dist = np.linalg.norm(relative_coords, 2, axis=1)
  selected_grids_coords = grid_coords[dist < dist_thres]
  
  grid_pose = np.identity(4)
  overlaps = []
  for selected_grids_coord in selected_grids_coords:
    new_x = str('{:+.2f}'.format(selected_grids_coord[0])).zfill(10)
    new_y = str('{:+.2f}'.format(selected_grids_coord[1])).zfill(10)
    file_name = new_x + '_' + new_y + '.npz'
    virtual_scan = np.load(os.path.join(virtual_scan_folder, file_name))['arr_0']
    grid_range = virtual_scan[:, :, 3]
    
    grid_num = len(grid_range[(grid_range > 0) & (grid_range <= 50)])
    
    grid_pose[0, 3] = selected_grids_coord[0]
    grid_pose[1, 3] = selected_grids_coord[1]
    grid_pose[2, 3] = current_pose[2, 3]
    
    visible_points = np.linalg.inv(grid_pose).dot(current_pose).dot(current_scan.T).T
    current_points_transformed = gen_virtual_scan(visible_points.astype(np.float32),
                                                  range_image_params['height'], range_image_params['width'],
                                                  range_image_params['fov_up'], range_image_params['fov_down'],
                                                  range_image_params['max_range'], range_image_params['min_range'])
    
    current_range_transformed = current_points_transformed[:, :, 3]
    
    overlap = np.count_nonzero(
      abs(grid_range[current_range_transformed > 0] -
          current_range_transformed[current_range_transformed > 0]) < 1) / min(valid_num, grid_num)
    
    overlaps.append([frame_idx, selected_grids_coord[0], selected_grids_coord[1], overlap])
  
  return overlaps


def com_overlaps(virtual_scan_folder, poses, scan_paths, overlap_file_path, range_image_params):
  """ Compute the ground truth overlap values for a sequence of LiDAR scans
    and generate a ground truth overlap file.
    Args:
      virtual_scan_folder: path of virtual scan folder
      poses: ground truth poses of the LiDAR scans
      scan_paths: paths of the LiDAR scans
      overlap_file_path: output file path of the ground truth overlaps
      range_image_params: parameters for generating a range image
  """
  # load virtual scans
  virtual_scan_paths = utils.load_files(virtual_scan_folder)
  
  grid_coords = []
  for virtual_scan_path in virtual_scan_paths:
    grid_coords.append(os.path.basename(virtual_scan_path).replace('.npz', '').split('_'))
  grid_coords = np.array(grid_coords, dtype=float)
  
  # ground truth format: each row contains [current_frame_idx, reference_frame_idx, overlap, yaw]
  print('generating raw overlap ground truth file...')
  if os.path.exists(overlap_file_path):
    print('the overlap mapping file already exists!')
  else:
    os.mkdir(os.path.dirname(overlap_file_path))
    ground_truth_overlap = []
    for idx in tqdm(range(len(poses))):
      overlaps = com_overlap(idx, grid_coords, virtual_scan_folder,
                             poses[idx], scan_paths[idx], range_image_params)
      if len(overlaps) > 0:
        ground_truth_overlap.append(overlaps)
    
    ground_truth_overlap = np.concatenate(ground_truth_overlap).reshape((-1, 4))
    np.savez_compressed(overlap_file_path, ground_truth_overlap)


if __name__ == '__main__':
  # load config file
  config_filename = '../../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # specify the output folders
  virtual_scan_folder = '../' + config['virtual_scan_folder']
  overlap_file_path = '../' + config['overlap_file_path']
  
  # load poses
  pose_file = '../' + config['pose_file']
  poses = utils.load_poses(pose_file)
  inv_frame0 = np.linalg.inv(poses[0])
  
  # load calibrations
  calib_file = '../' + config['calib_file']
  T_cam_velo = utils.load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)
  
  # convert kitti poses from camera coord to LiDAR coord
  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  poses = np.array(new_poses)
  
  # load LiDAR scans
  scan_folder = '../' + config['scan_folder']
  scan_paths = utils.load_files(scan_folder)
  
  range_image_params = config['range_image']
  
  com_overlaps(virtual_scan_folder, poses, scan_paths, overlap_file_path, range_image_params)
