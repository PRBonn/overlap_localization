#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script adds paths to the ground truth files which enables training using multiple sequences.

import os
import sys

import numpy as np
import yaml


def add_path_label(query_folder, map_folder, ground_truth_file, convert_tain_val=True):
  """ Add path to the training data according to the data structure.
    Args:
      query_folder: path of query folder.
      map_folder: path of map folder.
      ground_truth_file: path of ground truth file.
      convert_tain_val: also add path labels to the training and validation ground truth.
  """
  # load ground truth overlap and yaw
  npz_file = np.load(ground_truth_file)
  npz_file = npz_file['arr_0']
  
  seq_mapping = np.empty((npz_file.shape[0], 2), dtype=object)
  seq_mapping[:, 0] = query_folder
  seq_mapping[:, 1] = map_folder
  new_name = ground_truth_file
  np.savez_compressed(new_name, overlaps=npz_file, seq=seq_mapping)
  
  if convert_tain_val:
    train_file_name = os.path.join(os.path.dirname(ground_truth_file), 'train_set.npz')
    train_npz_file = np.load(train_file_name)
    train_npz_file = train_npz_file['arr_0']
    
    train_seq_mapping = np.empty((train_npz_file.shape[0], 2), dtype=object)
    train_seq_mapping[:, 0] = query_folder
    train_seq_mapping[:, 1] = map_folder
    np.savez_compressed(train_file_name, overlaps=train_npz_file, seq=train_seq_mapping)
    
    test_file_name = os.path.join(os.path.dirname(ground_truth_file), 'validation_set.npz')
    test_npz_file = np.load(test_file_name)
    test_npz_file = test_npz_file['arr_0']
    
    test_seq_mapping = np.empty((test_npz_file.shape[0], 2), dtype=object)
    test_seq_mapping[:, 0] = query_folder
    test_seq_mapping[:, 1] = map_folder
    np.savez_compressed(test_file_name, overlaps=test_npz_file, seq=test_seq_mapping)


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
  overlap_file_path = '../' + config['ground_truth']
  
  add_path_label('07/query', '07/training', overlap_file_path)
