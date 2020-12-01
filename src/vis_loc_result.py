#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script plots the final localization results and can visualize offline given the results.

import os
import sys
import yaml
import utils
import numpy as np
import matplotlib.pyplot as plt
from initialization import check_mapsize
from visualizer import Visualizer


def plot_traj_result(results, poses, numParticles=1000, grid_res=0.2, start_idx=0,
                     ratio=0.8, converge_thres=5, eva_thres=100):
  """ Plot the final localization trajectory.
    Args:
      results: localization results including particles in every timestamp.
      poses: ground truth poses.
      numParticles: number of particles.
      grid_res: the resolution of the grids.
      start_idx: the start index.
      ratio: the ratio of particles used to estimate the poes.
      converge_thres: a threshold used to tell whether the localization converged or not.
      eva_thres: a threshold to check the estimation results.
  """
  # get ground truth xy and yaw separately
  gt_location = poses[start_idx:, :2, 3]
  gt_heading = []
  for pose in poses:
    gt_heading.append(utils.euler_angles_from_rotation_matrix(pose[:3, :3])[2])
  gt_heading = np.array(gt_heading)[start_idx:]
  
  estimated_traj = []
  
  for frame_idx in range(start_idx, len(poses)):
    particles = results[frame_idx]
    # collect top 80% of particles to estimate pose
    idxes = np.argsort(particles[:, 3])[::-1]
    idxes = idxes[:int(ratio * numParticles)]
    
    partial_particles = particles[idxes]
    
    if np.sum(partial_particles[:, 3]) == 0:
      continue
      
    normalized_weight = partial_particles[:, 3] / np.sum(partial_particles[:, 3])

    estimated_traj.append(partial_particles[:, :3].T.dot(normalized_weight.T))

  estimated_traj = np.array(estimated_traj)
  
  # evaluate the results
  diffs_seperate = np.array(estimated_traj[:, :2] * grid_res - gt_location)
  diffs = np.linalg.norm(diffs_seperate, axis=1)  # diff in euclidean

  # check if every 100 success converged
  if np.all(diffs[eva_thres::eva_thres] < converge_thres):
    # calculate location error
    diffs_location = diffs[eva_thres:]
    mean_location = np.mean(diffs_location)
    mean_square_error = np.mean(diffs_location * diffs_location)
    rmse_location = np.sqrt(mean_square_error)

    # calculate heading error
    diffs_heading = np.minimum(abs(estimated_traj[eva_thres:, 2] - gt_heading[eva_thres:]),
                           2. * np.pi - abs(estimated_traj[eva_thres:, 2] - gt_heading[eva_thres:])) * 180. / np.pi
    mean_heading = np.mean(diffs_heading)
    mean_square_error_heading = np.mean(diffs_heading * diffs_heading)
    rmse_heading = np.sqrt(mean_square_error_heading)

    print('rmse_location: ', rmse_location)
    print('mean_location: ', mean_location)
    print('rmse_heading: ', rmse_heading)
    print('mean_heading: ', mean_heading)
  
  # plot results
  fig = plt.figure(figsize=(16, 10))
  ax = fig.add_subplot(111)

  ax.plot(poses[:, 0, 3], poses[:, 1, 3], c='r', label='ground_truth')
  ax.plot(estimated_traj[:, 0] * grid_res, estimated_traj[:, 1] * grid_res, label='weighted_mean_80%')
  plt.show()


def vis_offline(results, poses, map_poses, mapsize, numParticles=1000, grid_res=0.2, start_idx=0):
  """ Visualize localization results offline.
    Args:
      results: localization results including particles in every timestamp.
      poses: ground truth poses.
      map_poses: poses used to generate the map.
      mapsize: size of the map.
      numParticles: number of particles.
      grid_res: the resolution of the grids.
      start_idx: the start index.
  """
  plt.ion()
  visualizer = Visualizer(mapsize, poses, map_poses,
                          numParticles=numParticles,
                          grid_res=grid_res, strat_idx=start_idx)
  for frame_idx in range(start_idx, len(poses)):
    particles = results[frame_idx]
    visualizer.update(frame_idx, particles)
    visualizer.fig.canvas.draw()
    visualizer.fig.canvas.flush_events()


if __name__ == '__main__':
  # load config file
  config_filename = '../config/localization.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  config = yaml.load(open(config_filename), Loader=yaml.FullLoader)

  start_idx = config['start_index']
  grid_res = config['resolution']
  numParticles = config['numParticles']
  save_result = config['save_result']
  visualize = config['visualize']
  data_root_folder = config['data_root_folder']

  # load poses
  pose_file = config['pose_file']
  poses = utils.load_poses(pose_file)
  inv_frame0 = np.linalg.inv(poses[0])

  # load calibrations
  calib_file = config['calib_file']
  T_cam_velo = utils.load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)

  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  poses = np.array(new_poses)

  # load map
  map_folder = os.path.join(data_root_folder, config['infer_seqs_map'], 'feature_volumes')
  mapsize, grid_coords = check_mapsize(map_folder, grid_res)

  # load results
  result_file = 'localization_results_' + str(start_idx) + '.npz'
  if os.path.exists(result_file):
    results = np.load(result_file)['arr_0']
  else:
    print('result file does not exists at: ', result_file)
    exit(-1)
  
  # test trajectory plotting
  plot_traj_result(results, poses, numParticles=numParticles, grid_res=grid_res, start_idx=start_idx)
  
  # test offline visualizer
  vis_offline(results, poses, poses, mapsize,
              numParticles=numParticles, grid_res=grid_res, start_idx=start_idx)