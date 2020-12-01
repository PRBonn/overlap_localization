#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script contains the fast inferring class which is used for online overlap and yaw estimations.

import os
import sys
import numpy as np

from FeatureVolumeCacheSequence import FeatureVolumeCacheSequence
sys.path.append('../OverlapNet/src/two_heads')
from ImagePairOverlapOrientationSequence import ImagePairOverlapOrientationSequence
from infer import Infer


class FastInfer(Infer):
  """ This is a class for fast online OverlapNet inferring with multiple frames
  """
  def __init__(self, config, cache_size):
    """
      config: configure parameters.
      cache_size: number of cached feature volumes.
    """
    if not 'infer_seqs' in config:
      # Assuming query sequence (first leg) AND map_sequence (second leg)
      # super.self.seq is the query, self.seq_map is the label of the map
      config['infer_seqs']=config['infer_seqs_query']
      self.seq_map=config['infer_seqs_map']
      
    super().__init__(config)
    feature_volume_size = (int(self.head.input_shape[0][1]),
                           int(self.head.input_shape[0][2]),
                           int(self.head.input_shape[0][3]))
    self.volume_cache = FeatureVolumeCacheSequence(config, feature_volume_size, cache_size)
  
  def infer_multiple(self, idx_current_frame, coordinates_nearby_grid):
    """
      idx_current_frame: current query scan index.
      coordinates_nearby_grid: coordinates of grids assigned to particles.
    """
    self.volume_cache.new_task(idx_current_frame, coordinates_nearby_grid)
    
    model_outputs = self.head.predict_generator(self.volume_cache, max_queue_size=10,
                                                workers=1, verbose=1)
    # in case of single head, make output a list of size 1                                     
    if not isinstance(model_outputs, list):
      model_outputs = [model_outputs]
    
    return model_outputs
  
  def print_statistics(self):
    self.volume_cache.print_statistics()
  
  def coord2filename(self, coord):
    """
      Args: 1x2 numpy array of X and Y coordinate.
      Returns: complete filename for the feature volume.
    """
    new_x = str('{:+.2f}'.format(coord[0])).zfill(10)
    new_y = str('{:+.2f}'.format(coord[1])).zfill(10)
    file_name = new_x + '_' + new_y
    return file_name
  
  def coord_or_idx2filename(self, coord_or_idx):
    """
      Args: A numpy array of X and Y coordinate
        or an index as int (numpy.integer type).
      Returns: filename (without any extension).
    """
    if type(coord_or_idx)==np.ndarray:
      new_x = str('{:+.2f}'.format(coord_or_idx[0])).zfill(10)
      new_y = str('{:+.2f}'.format(coord_or_idx[1])).zfill(10)
      file_name = new_x + '_' + new_y
    else:
      file_name = str(coord_or_idx).zfill(6)
      
    return file_name

  def save_feature_volumes(self, coords_or_idx, save_new_volumes=True):
    """ For external usage to save the feature volumes.
      Input:
        coords_or_idx: Either nx2 numpy array of map coordinates X,Y
          or 1D array of size n numpy array of frame indices (thus filenames will be e.g. 000000.npy).
    """
    n = len(coords_or_idx)
    filenames_for_generation = []
    coordinate_for_generation = []
    feature_volumes_exist = []
    is_new = np.ones(n)
    
    for i in range(n):
      filename = self.coord_or_idx2filename(coords_or_idx[i])
      complete_path = self.datasetpath + '/feature_volumes/' + \
                      filename + '.npz'
      
      if os.path.exists(complete_path):
        feature_volumes_exist.append(np.load(complete_path)['arr_0'])
        is_new[i] = 0
      else:
        filenames_for_generation.append(filename)
        coordinate_for_generation.append(coords_or_idx[i])
        if coords_or_idx.ndim==2:
          print("Generate new feature volume for (%f, %f)" % (coords_or_idx[i, 0], coords_or_idx[i, 1]))
        else:
          print("Generate new feature volume for index %d" % coords_or_idx[i])

    feature_volumes_exist = np.array(feature_volumes_exist)
    
    generation_size = len(filenames_for_generation)
    if generation_size > 0:
      if generation_size % self.batch_size == 0:
        loop_range = int(generation_size / self.batch_size)
      else:
        loop_range = int(generation_size / self.batch_size) + 1
      for loop_idx in range(loop_range):
        if (generation_size - loop_idx * self.batch_size) < self.batch_size:
          batch_end = generation_size
        else:
          batch_end = (loop_idx + 1) * self.batch_size
        real_batch_size = len(filenames_for_generation[loop_idx * self.batch_size:batch_end])
        test_generator_leg = ImagePairOverlapOrientationSequence(self.datasetpath,
                                                                 filenames_for_generation[
                                                                 loop_idx * self.batch_size:batch_end],
                                                                 [],
                                                                 [self.seq for _ in
                                                                  range(len(filenames_for_generation))],
                                                                 [],
                                                                 np.zeros((real_batch_size)),
                                                                 np.zeros((real_batch_size)),
                                                                 self.network_output_size, self.batch_size,
                                                                 self.inputShape[0], self.inputShape[1],
                                                                 self.no_input_channels,
                                                                 use_depth=self.use_depth,
                                                                 use_normals=self.use_normals,
                                                                 use_class_probabilities=self.use_class_probabilities,
                                                                 use_intensity=self.use_intensity,
                                                                 use_class_probabilities_pca=self.use_class_probabilities_pca)
        
        feature_volumes_new = self.leg.predict_generator(test_generator_leg, max_queue_size=10,
                                                         workers=8, verbose=1)
        
        if save_new_volumes:
          for i in range(len(feature_volumes_new)):
            f = self.datasetpath + '/' + self.seq + '/feature_volumes/'
            f += self.coord_or_idx2filename(coordinate_for_generation[i + loop_idx * self.batch_size])
            if coords_or_idx.ndim!=2:
              f += '.npz'

            np.savez_compressed(f, feature_volumes_new[i])


# Test code    
if __name__ == '__main__':
  pass
