#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the main script file for overlap-based Monte Carlo localization.

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

import utils

from initialization import check_mapsize, init_particles_given_coords
from motion_model import motion_model, gen_commands
from sensor_model_overlap import SensorModel
from resample import resample

from visualizer import Visualizer
from vis_loc_result import plot_traj_result
  
if __name__ == '__main__':
  # load config file
  config_filename = '../config/localization.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]

  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))  
  
  # setup parameters
  start_idx = config['start_index']
  grid_res = config['resolution']
  numParticles = config['numParticles']
  save_result = config['save_result']
  visualize = config['visualize']
  data_root_folder = config['data_root_folder']
  seq_idx_map = config['infer_seqs_map']
  seq_idx_query = config['infer_seqs_query']
  move_thres = config['move_thres']
  
  # load map
  map_folder = os.path.join(data_root_folder, seq_idx_map, 'feature_volumes')
  mapsize, grid_coords = check_mapsize(map_folder, grid_res)
  
  # load poses
  pose_file = config['pose_file']
  poses = utils.load_poses(pose_file)
  inv_frame0 = np.linalg.inv(poses[0])
  
  # load calibrations
  calib_file = config['calib_file']
  T_cam_velo = utils.load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)
  
  # convert poses in LiDAR coordinate system
  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  new_poses = np.array(new_poses)
  poses = new_poses

  # initialize sensor model
  sensor_model = SensorModel(config, mapsize, map_folder)

  # generate motion commands
  commands = gen_commands(poses, grid_res)

  # initialize particles
  particles = init_particles_given_coords(numParticles, grid_coords)
  is_initial = True
  
  if visualize:
    plt.ion()
    visualizer = Visualizer(mapsize, poses, poses, strat_idx=start_idx)
    
  if save_result:
    loc_results = np.empty((len(poses), numParticles, 4))
  
  for frame_idx in range(start_idx, len(poses)):
    if visualize:
      visualizer.update(frame_idx, particles)
      visualizer.fig.canvas.draw()
      visualizer.fig.canvas.flush_events()
    
    # motion model
    particles = motion_model(particles, commands[frame_idx])

    # only update the weight when the car moves
    if commands[frame_idx, 1] > 0.2 / grid_res or is_initial:
      is_initial = False
      
      # grid-based method
      particles = sensor_model.update_weights(particles, frame_idx)
      
      # resampling
      particles = resample(particles)
    
    if save_result:
      loc_results[frame_idx, :len(particles)] = particles
    
    print('finished frame:', frame_idx)

  if save_result:
    print('Saving localization results...')
    np.savez_compressed('localization_results_'+str(start_idx), loc_results)
    plot_traj_result(loc_results, poses, numParticles=numParticles, start_idx=start_idx)