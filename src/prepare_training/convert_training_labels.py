#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script convert the ground truth file into overlapnet format
# Input: ground truth file, [current_frame_idx, x_coord, y_coord, overlap], x_coord, y_coord are grid coordinates
# Output: OverlapNet training format: [current_frame_idx, reference_frame_idx, overlap, yaw_angle, x_coord, y_coord]

import sys
import yaml
import numpy as np
from tqdm import tqdm

import utils

pi = np.pi


def convert_training_labels(overlap_file,
                            overlap_yaw_file_overlapnet_format,
                            rename_lut_file,
                            poses, grid_res=0.2, save_rename_lut=True):
  """ Convert the training ground truth into OverlapNet format.
    Args:
      overlap_file: raw ground truth overlap file.
      overlap_yaw_file_overlapnet_format: the output name of converted ground truth file.
      rename_lut_file: the file name of the renaming lookup table.
      poses: ground truth poses.
      grid_res: the resolution of the grids.
      save_rename_lut: whether to save the renaming lookup table.
  """
  # load overlap labels
  raw_overlaps = np.load(overlap_file)['arr_0'].astype('float32')
  
  # init yaw labels
  yaw_idxs = []
  yaw_resolution = 360  # depend on the net structure, equal to the size of last layer output
  
  # create fake indexes for all grid frames
  reference_idxs = []
  # use a lookup table to avoid naming multiple times
  grid_x_coord_max = round(np.max(raw_overlaps[:, 1]) / grid_res)
  grid_x_coord_min = round(np.min(raw_overlaps[:, 1]) / grid_res)
  grid_y_coord_max = round(np.max(raw_overlaps[:, 2]) / grid_res)
  grid_y_coord_min = round(np.min(raw_overlaps[:, 2]) / grid_res)
  
  grid_x_size = int(grid_x_coord_max - grid_x_coord_min + 1)
  grid_y_size = int(grid_y_coord_max - grid_y_coord_min + 1)
  
  rename_lut = np.full((grid_y_size, grid_x_size), -1, dtype=int)
  new_idx = 0
  print('Converting ground truth labels into OverlapNet format...')
  for idx in tqdm(range(len(raw_overlaps))):
    current_idx = int(raw_overlaps[idx, 0])
    grid_x_coord = round(raw_overlaps[idx, 1] / grid_res)
    grid_y_coord = round(raw_overlaps[idx, 2] / grid_res)
    
    # create the fake idx for grid frames
    if rename_lut[int(grid_y_coord_max - grid_y_coord), int(grid_x_coord - grid_x_coord_min)] >= 0:
      refrence_idx = rename_lut[int(grid_y_coord_max - grid_y_coord), int(grid_x_coord - grid_x_coord_min)]
    else:
      refrence_idx = new_idx
      rename_lut[int(grid_y_coord_max - grid_y_coord), int(grid_x_coord - grid_x_coord_min)] = refrence_idx
      new_idx += 1
    reference_idxs.append(refrence_idx)
    
    current_pose = poses[current_idx]
    
    current_rotation = current_pose[:3, :3]
    _, _, yaw = utils.euler_angles_from_rotation_matrix(current_rotation)
    # convert to OverlapNet training format
    yaw_element_idx = int(- (yaw / pi) * yaw_resolution // 2 + yaw_resolution // 2)
    yaw_idxs.append(yaw_element_idx)
  
  overlaps_yaws = np.zeros((raw_overlaps.shape[0], raw_overlaps.shape[1] + 2))
  overlaps_yaws[:, 0] = raw_overlaps[:, 0]  # current frame idx
  overlaps_yaws[:, 1] = np.array(reference_idxs)  # reference frame idx
  overlaps_yaws[:, 2] = raw_overlaps[:, 3]  # overlaps
  overlaps_yaws[:, 3] = np.array(yaw_idxs)  # yaw angles
  overlaps_yaws[:, 4] = raw_overlaps[:, 1]  # grid x coord
  overlaps_yaws[:, 5] = raw_overlaps[:, 2]  # grid y coord
  
  np.savez_compressed(overlap_yaw_file_overlapnet_format, overlaps_yaws)
  
  if save_rename_lut:
    np.savez_compressed(rename_lut_file, rename_lut)
    print('saved the renaming look up table')


if __name__ == '__main__':
  # load config file
  config_filename = '../../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]

  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
   
  # specify file names
  overlap_file_path = config['overlap_file_path']
  overlap_yaw_ground_truth_path = config['ground_truth']
  rename_lut_path = config['rename_lut']  
  
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
  
  convert_training_labels(overlap_file_path, overlap_yaw_ground_truth_path, rename_lut_path, poses)
