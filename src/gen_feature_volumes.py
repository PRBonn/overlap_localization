#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this script generates feature volumes for overlap-based Monte Carlo localization.

import os
import sys
import numpy as np
import yaml
from fast_infer import FastInfer


def gen_feature_volumes_map(config, cache_size=50000):
  """ Generate feature volumes for grids.
    Args:
      config: configuration parameters.
  """
  # create directory for feature volumes if not there
  features_folder = os.path.join(config['data_root_folder'], config['infer_seqs_map'], 'feature_volumes')
  if not os.path.exists(features_folder):
    os.mkdir(features_folder)

  # init fast infer
  config['infer_seqs']=config['infer_seqs_map']
  infer = FastInfer(config, cache_size=cache_size)
  
  # collect coords
  depth_folder = os.path.join(config['data_root_folder'], config['infer_seqs_map'], 'depth')
  grid_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(depth_folder)) for f in fn]
  grid_paths.sort()
  
  coords = []
  for grid_path in grid_paths:
    coords.append(os.path.basename(grid_path).replace('.npy', '').split('_'))
  
  coords = np.array(coords, dtype=float)
  
  infer.save_feature_volumes(coords)


def gen_feature_volumes_query(config, cache_size=50000):
  """ Generate feature volumes for query scans.
    Args:
      config: configuration parameters.
  """
  # create directory for feature volumes if not there
  features_folder = os.path.join(config['data_root_folder'], config['infer_seqs_query'], 'feature_volumes')
  if not os.path.exists(features_folder):
    os.mkdir(features_folder)
  
  # init fast infer
  config['infer_seqs']=config['infer_seqs_query']
  infer = FastInfer(config, cache_size=cache_size)
  
  # collect indizes
  depth_folder = os.path.join(config['data_root_folder'], config['infer_seqs_query'], 'depth')
  idx_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(depth_folder)) for f in fn]
  idx_paths.sort()
  
  idxs = []
  for idx_path in idx_paths:
    idxs.append(os.path.basename(idx_path).replace('.npy', ''))
  
  idxs = np.array(idxs, dtype=int)
  
  # save features
  infer.save_feature_volumes(idxs)


if __name__ == '__main__':
  # load overlapnet model
  config_filename = '../config/localization.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]

  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  print('==============================================================================')
  print('Generate feature volumes for map ...')
  gen_feature_volumes_map(config)

  print(' ')
  print('==============================================================================')
  print('Generate feature volumes for queries ...')
  gen_feature_volumes_query(config)
  
  
