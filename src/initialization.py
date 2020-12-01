#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: some functions for MCL initialization

import os
import numpy as np

np.random.seed(0)


def init_particles_uniform(map_size, numParticles):
  """ Initialize particles uniformly.
    Args:
      map_size: size of the map.
      numParticles: number of particles.
    Return:
      particles.
  """
  [x_min, x_max, y_min, y_max] = map_size
  particles = []
  rand = np.random.rand
  for i in range(numParticles):
    x = (x_max - x_min) * rand(1) + x_min
    y = (y_max - y_min) * rand(1) + y_min
    # theta = 2 * np.pi * rand(1)
    theta = -np.pi + 2 * np.pi * rand(1)
    weight = 1
    particles.append([x, y, theta, weight])
  
  return np.array(particles)


def init_particles_given_coords(numParticles, coords, init_weight=1.0):
  """ Initialize particles uniformly given the road coordinates.
    Args:
      numParticles: number of particles.
      coords: road coordinates
    Return:
      particles.
  """
  particles = []
  rand = np.random.rand
  args_coords = np.arange(len(coords))
  selected_args = np.random.choice(args_coords, numParticles)
  
  for i in range(numParticles):
    x = coords[selected_args[i]][0]
    y = coords[selected_args[i]][1]
    # theta = 2 * np.pi * rand(1)
    theta = -np.pi + 2 * np.pi * rand(1)
    particles.append([x, y, theta, init_weight])
  
  return np.array(particles, dtype=float)


def check_mapsize(map_folder, grid_res=0.2):
  """ Compute the size of the map.
    Args:
      map_folder: the path of the map folder.
      grid_res: the resolution of the grids.
    Return:
      size of the map and the road coordinates.
  """
  virtual_scan_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(map_folder)) for f in fn]
  virtual_scan_paths.sort()
  virtual_scan_paths = np.array(virtual_scan_paths)
  grid_coords = []
  
  filename, file_extension = os.path.splitext(virtual_scan_paths[0])
  
  if '.png' in file_extension:
    for virtual_scan_path in virtual_scan_paths:
      coord = os.path.basename(virtual_scan_path).replace('.png', '').split('_')
      if len(coord) > 1:
        grid_coords.append(coord)
  else:
    for virtual_scan_path in virtual_scan_paths:
      coord = os.path.basename(virtual_scan_path).replace('.npz', '').split('_')
      if len(coord) > 1:
        grid_coords.append(coord)
  
  grid_coords = np.array(grid_coords, dtype=float)
  grid_coords = grid_coords[grid_coords[:, 0] < 1000]  # get rid of the fake current frame idx
  
  # to make the grid coords as integers
  grid_coords = grid_coords / grid_res
  
  offset = 1  # meter
  min_x = int(np.round(np.min(grid_coords[:, 0])))
  max_x = int(np.round(np.max(grid_coords[:, 0])))
  min_y = int(np.round(np.min(grid_coords[:, 1])))
  max_y = int(np.round(np.max(grid_coords[:, 1])))
  # print('[min_x, max_x, min_y, max_y]: ', [min_x, max_x, min_y, max_y])
  
  return [min_x, max_x, min_y, max_y], grid_coords
