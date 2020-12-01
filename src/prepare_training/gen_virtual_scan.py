#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates virtual scans for map grids using a voxelized point cloud map

import os
import sys
import yaml
import numpy as np
import open3d as o3d
from tqdm import tqdm

import utils

try:
  from c_gen_virtual_scan import gen_virtual_scan
except:
  print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
  sys.exit(-1)


def gen_pcd_map(poses, scan_paths, map_file,
                voxel_size=0.02, max_dist=50,
                min_dist=3, min_z=-2, vis_map=False):
  """ Generate a global point cloud map.
    Args:
      poses: ground truth poses.
      scan_paths: paths of LiDAR scans.
      map_file: output path of the global point cloud map.
    Returns:
      pcd_map: the global point cloud map in open3d format.
  """
  pcd_map = o3d.geometry.PointCloud()
  for idx in tqdm(range(len(scan_paths))):
    curren_points = utils.load_vertex(scan_paths[idx])
    dist = np.linalg.norm(curren_points[:, :3], 2, axis=1)
    curren_points = curren_points[(dist < max_dist) &
                                  (dist > min_dist) &
                                  (curren_points[:, 2] > min_z)]
    localcloud = o3d.geometry.PointCloud()
    localcloud.points = o3d.utility.Vector3dVector(curren_points[:, :3])
    localcloud.transform(poses[idx])
    pcd_map += localcloud
  
  print('Downsampling with voxel size of: ', voxel_size)
  pcd_map = pcd_map.voxel_down_sample(voxel_size)
  o3d.io.write_point_cloud(map_file, pcd_map)
  print('Finished and saved the map in: ', map_file)
  
  # visualize pcd map
  if vis_map:
    o3d.visualization.draw_geometries([pcd_map])
  
  return pcd_map


def gen_grid(virtual_scan_folder, file_name, point_cloud_points, range_image_params):
  """ Generate virtual scan for each grid.
    Args:
      virtual_scan_folder: path of virtual scan folder.
      file_name: file name of the virtual scan.
      point_cloud_points: local point clouds used to generate the virtual scan.
      range_image_params: parameters for generating a range image.
  """
  # generate depth image
  virtual_scan = gen_virtual_scan(point_cloud_points.astype(np.float32),
                                  range_image_params['height'], range_image_params['width'],
                                  range_image_params['fov_up'], range_image_params['fov_down'],
                                  range_image_params['max_range'], range_image_params['min_range'])
  
  # save virtual scan
  np.savez_compressed(os.path.join(virtual_scan_folder, file_name), virtual_scan)


def crop_cloud_with_bbox(cloud, center=[0, 0], length=50, width=50, height=5):
  """ Crop the global point cloud.
    Args:
      cloud: the global point cloud.
    Returns:
      a cropped point cloud.
  """
  bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(center[0] - length, center[1] - width, -height),
                                             max_bound=(center[0] + length, center[1] + width, +height))
  return cloud.crop(bbox)


def rasterize_map(poses, pcd_map, virtual_scan_folder, grid_res, offset, range_image_params):
  """ Rasterize the global map into grids.
    Args:
      poses: ground truth poses.
      pcd_map: the global point cloud map
      virtual_scan_folder: path of virtual scan folder.
      grid_res: the resolution of the grids.
      offset: the offset of the border
      range_image_params: parameters for generating a range image.
  """
  # check the virtual_scan_folder
  if not os.path.exists(virtual_scan_folder):
    os.makedirs(virtual_scan_folder)
  
  # initialize a road look up table
  xyzs = poses[:, :3, 3]
  min_x = int(np.round((np.min(xyzs[:, 0]) - offset) / grid_res))
  max_x = int(np.round((np.max(xyzs[:, 0]) + offset) / grid_res))
  min_y = int(np.round((np.min(xyzs[:, 1]) - offset) / grid_res))
  max_y = int(np.round((np.max(xyzs[:, 1]) + offset) / grid_res))
  lut_x_size = (max_y - min_y) + 1
  lut_y_size = (max_x - min_x) + 1  # use (0,0) as origin
  grid_map_lut = np.full((lut_x_size, lut_y_size), -1, dtype=np.float32)
  
  # local grid coordinates
  loc_coords = []
  for x_coord in np.arange(-offset, offset + grid_res, grid_res):
    for y_coord in np.arange(-offset, offset + grid_res, grid_res):
      loc_coords.append([x_coord, y_coord])
  loc_coords = np.array(loc_coords)
  
  print('start generating virtual scans...')
  
  # fill in the road lookup table
  for frame_idx in tqdm(range(len(xyzs))):
    for loc_coord in loc_coords:
      grid_pose = np.identity(4)
      # covert local grid coordinates to global coordinates with rounding
      x_global = round((xyzs[frame_idx, 0] + loc_coord[0]) / grid_res) * grid_res
      y_global = round((xyzs[frame_idx, 1] + loc_coord[1]) / grid_res) * grid_res
      
      # check road lut
      lut_x = int(round(max_y - y_global / grid_res))
      lut_y = int(round(x_global / grid_res - min_x))
      if lut_x < 0 or lut_x >= lut_x_size or lut_y < 0 or lut_y >= lut_y_size:
        continue
        
      if grid_map_lut[lut_x, lut_y] < 1:
        grid_map_lut[lut_x, lut_y] = 1
        
        new_x = str('{:+.2f}'.format(x_global)).zfill(10)
        new_y = str('{:+.2f}'.format(y_global)).zfill(10)
        file_name = new_x + '_' + new_y
        
        # check existence
        if os.path.exists(os.path.join(virtual_scan_folder, file_name + '.npz')):
          print('existing: ', file_name)
          continue
        
        pcd_map_tmp = pcd_map
        pcd_map_tmp = crop_cloud_with_bbox(pcd_map_tmp, center=[x_global, y_global])
        
        grid_pose[0, 3] = x_global
        grid_pose[1, 3] = y_global
        grid_pose[2, 3] = xyzs[frame_idx, 2]
        current_points = np.array(pcd_map_tmp.points)
        homo_points = np.ones((current_points.shape[0], current_points.shape[1] + 1), dtype=np.float32)
        homo_points[:, :-1] = current_points
        homo_points = np.linalg.inv(grid_pose).dot(homo_points.T).T
        
        gen_grid(virtual_scan_folder, file_name, homo_points, range_image_params)


if __name__ == '__main__':
  # load config file
  config_filename = '../../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # specify parameters
  resolution = config['resolution']  # resolution of grid is 20 cm
  offset = config['offset']  # for each frame, we generate grids inside a 1m*1m square
  num_frames = config['num_frames']
  
  # specify the output folders
  virtual_scan_folder = '../' + config['virtual_scan_folder']
  map_file = '../' + config['map_file']
  
  # load poses
  pose_file = '../' + config['pose_file']
  poses = np.array(utils.load_poses(pose_file))
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
  scan_folder = config['scan_folder']
  scan_paths = utils.load_files(scan_folder)
  
  # test for the first N scans
  if num_frames >= len(poses) or num_frames <= 0:
    print('generate training data for all frames with number of: ', len(poses))
  else:
    poses = poses[:num_frames]
    scan_paths = scan_paths[:num_frames]
  
  range_image_params = config['range_image']
  
  # step1: build the pcd map
  if os.path.exists(map_file):
    pcd_map = o3d.io.read_point_cloud(map_file)
    num_points = len(pcd_map.points)
    if num_points > 0:
      print('Successfully load pcd map with point size of: ', num_points)
  else:
    print('Creating a pcd map...')
    pcd_map = gen_pcd_map(poses, scan_paths, map_file, vis_map=False)
  
  # step2: generate virtual scans
  rasterize_map(poses, pcd_map, virtual_scan_folder, resolution, offset, range_image_params)
