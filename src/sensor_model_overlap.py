#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the sensor model for overlap-based Monte Carlo localization.
#        This model use grid map, where each grid contains a virtual frame.
import os
import numpy as np
import matplotlib.pyplot as plt
from fast_infer import FastInfer


class SensorModel():
  """ This class is the implementation of using overlap predictions from OverlapNet
    as the sensor model for localization. In this sensor model we discretize the environment and generate a virtual
    frame for each grid after discretization. We use OverlapNet estimate the overlaps between the current frame and
    the grid virtual frames and use the predictions as the observation measurement.
  """
  def __init__(self, config, mapsize, map_folder):
    """ initialization:
      config_file: the configuration file of the OverlapNet
      mapsize: the size of the given map
      map_folder: the folder contains the feature volume map
    """
    # because we round the coordinates of particles, therefore it is safer to have an offset to the border
    self.offset = 1
    self.x_min = round(mapsize[0])
    self.x_max = round(mapsize[1])
    self.y_min = round(mapsize[2])
    self.y_max = round(mapsize[3])
    # create a grid files lookup table
    self.grid_paths_lut = np.full((self.y_max - self.y_min + self.offset,
                                   self.x_max - self.x_min + self.offset), 0, dtype=np.object)
    
    # map resolution
    self.resolution = config['resolution']

    # initialize fast infer
    self.model = FastInfer(config, cache_size=50000)
    
    # for check the file existing
    self.map_folder = map_folder
    
    # get grid coords
    self.coords = []
    feature_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(self.map_folder)) for f in fn]
    feature_paths.sort()
    feature_paths = np.array(feature_paths)
    for feature_path in feature_paths:
      self.coords.append(os.path.basename(feature_path).replace('.npz', '').split('_'))
    self.coords = np.array(self.coords, dtype=float)
    
    # check whether correct yaw angle
    self.use_yaw = config['use_yaw']
    self.yaw_sigma = config['yaw_sigma'] * np.pi / 180.
      
    self.is_converged = False
    self.num_reduced = config['num_reduced']
    self.converge_thres = config['converge_thres']
    self.min_overlap_for_angle = config['min_overlap_for_angle']
    
    # parameters for weight updating
    self.default_weight = 0.1
    self.invalid_weight = 0.001

  def update_weights(self, particles, frame_idx):
    """ This function update the weight for each particle using batch.
      Args:
        particles: each particle has four properties [x, y, theta, weight]
        measurements: overlap heatmaps
      Returns:
        particles ... same particles with changed particles(i).weight
    """
  
    # lookup table for overlap, for every current frame the lookup tables are different
    # We therefore initialize it for every current frame
    overlap_lut = np.full((self.y_max - self.y_min + 1, self.x_max - self.x_min + 1), -1, dtype=np.float32)
    new_particle = particles
    infer_coords = []
    overlap_idxes = 0

    # first collect the grid indexes to calculate overlaps
    for idx in range(len(particles)):
      particle = particles[idx]
      x = int(round(particle[0]))
      y = int(round(particle[1]))

      # check whether there is feature volume
      real_x = x * self.resolution
      real_y = y * self.resolution
      new_x = str('{:+.2f}'.format(real_x)).zfill(10)
      new_y = str('{:+.2f}'.format(real_y)).zfill(10)
      file_name = new_x + '_' + new_y + '.npz'

      path = os.path.join(self.map_folder, file_name)
      if not os.path.exists(path):
        continue

      # check a grid sampled or not
      check_flag = overlap_lut[int(self.y_max - y), int(x - self.x_min)]
      if check_flag < 0:
        infer_coords.append([real_x, real_y])
        overlap_lut[int(self.y_max - y), int(x - self.x_min)] = overlap_idxes
        overlap_idxes += 1

    # if no new inferring, skip the weight updating
    if len(infer_coords) == 0:
      return particles
    
    # inferring overlaps
    infer_coords = np.array(infer_coords)
    results_overlapnet = self.model.infer_multiple(frame_idx, infer_coords)
    overlaps = results_overlapnet[0]
    if self.use_yaw:
      yaws = np.argmax(results_overlapnet[1], axis=1)
      yaws = - (yaws - 180.) * np.pi / 180.  # convert from OverlapNet output to real yaw
    
    # update particle weights
    all_overlaps = np.ones(len(particles)) * self.default_weight
    all_yaws = np.ones(len(particles)) * self.default_weight
    for idx in range(len(particles)):
      particle = particles[idx]
      if particle[0] < self.x_min + self.offset or particle[0] > self.x_max - self.offset \
        or particle[1] < self.y_min + self.offset or particle[1] > self.y_max - self.offset:
        overlap = self.invalid_weight
      else:
        # update overlap
        x = int(round(particle[0]))
        y = int(round(particle[1]))
        overlap_idx = overlap_lut[int(self.y_max - y), int(x - self.x_min)]
        if overlap_idx < 0:
          continue
        overlap = overlaps[int(overlap_idx)]
        if self.use_yaw:
          if overlap >= self.min_overlap_for_angle:
            delta_yaw = min(abs(yaws[int(overlap_idx)] - particle[2]),
                            2*np.pi - abs(yaws[int(overlap_idx)] - particle[2]))
            all_yaws[idx] = np.exp(-0.5 * delta_yaw * delta_yaw / (self.yaw_sigma * self.yaw_sigma))
        
      all_overlaps[idx] = overlap

    # update the weights of the particles
    if self.use_yaw:
      # update weight also use yaw angle estimation
      new_particle[:, 3] = new_particle[:, 3] * all_overlaps * all_yaws
    else:
      new_particle[:, 3] = new_particle[:, 3] * all_overlaps

    # check convergence using the number of occupied grids
    occupied_grid = overlap_lut[overlap_lut > 0]
    if len(occupied_grid) < self.converge_thres and not self.is_converged:
      self.is_converged = True
      print('Converged!')
  
      idxes = np.argsort(new_particle[:, 3])[::-1]
      if not self.num_reduced > len(new_particle) or self.num_reduced < 0:
        new_particle = new_particle[idxes[:self.num_reduced]]
        
    # normalization
    new_particle[:, 3] = new_particle[:, 3] / np.max(new_particle[:, 3])
  
    return new_particle
    
  def save_error_map(self, error_map, frame_idx):
    """ This function generate error maps,
      which is used to visualize prediction results for each step.
    """
    fig0, ax0 = plt.subplots(figsize=(20, 20))
    # ax0.imshow(overlaps, cmap='hot')
    ax0.imshow(error_map)
    
    ax0.set_xticks(np.arange(0, self.x_max - self.x_min, 10))
    ax0.set_yticks(np.arange(0, self.y_max - self.y_min, 10))
    ax0.set_xticklabels(np.arange(self.x_min, self.x_max+10, 10).astype(str))
    ax0.set_yticklabels(np.arange(self.y_min, self.y_max+10, 10)[::-1].astype(str))
    # ax0.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    
    ax0.set_title('test')
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    
    fig0.tight_layout()
    # plt.show()
    fig0.savefig('test/' + str(frame_idx).zfill(6))
    plt.close()


if __name__ == '__main__':
  pass
