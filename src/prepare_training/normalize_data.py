#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a simple example to normalize the overlap data, one could do the same to yaw

import os
import sys
import yaml
import numpy as np


def normalize_data(ground_truth_file):
  """ Normalize the training data according to the overlap value.
     Args:
       ground_truth_mapping: the raw ground truth mapping array.
     Returns:
       dist_norm_data: normalized ground truth mapping array.
  """

  ground_truth_mapping = np.load(ground_truth_file)['arr_0']
  
  gt_map = ground_truth_mapping
  bin_0_9 = gt_map[np.where(gt_map[:, 3] < 0.1)]
  bin_10_19 = gt_map[(gt_map[:, 3] < 0.2) & (gt_map[:, 3] >= 0.1)]
  bin_20_29 = gt_map[(gt_map[:, 3] < 0.3) & (gt_map[:, 3] >= 0.2)]
  bin_30_39 = gt_map[(gt_map[:, 3] < 0.4) & (gt_map[:, 3] >= 0.3)]
  bin_40_49 = gt_map[(gt_map[:, 3] < 0.5) & (gt_map[:, 3] >= 0.4)]
  bin_50_59 = gt_map[(gt_map[:, 3] < 0.6) & (gt_map[:, 3] >= 0.5)]
  bin_60_69 = gt_map[(gt_map[:, 3] < 0.7) & (gt_map[:, 3] >= 0.6)]
  bin_70_79 = gt_map[(gt_map[:, 3] < 0.8) & (gt_map[:, 3] >= 0.7)]
  bin_80_89 = gt_map[(gt_map[:, 3] < 0.9) & (gt_map[:, 3] >= 0.8)]
  bin_90_100 = gt_map[(gt_map[:, 3] <= 1) & (gt_map[:, 3] >= 0.9)]

  # # print the distribution
  # distribution = [len(bin_0_9), len(bin_10_19), len(bin_20_29), len(bin_30_39), len(bin_40_49),
  #                 len(bin_50_59), len(bin_60_69), len(bin_70_79), len(bin_80_89), len(bin_90_100)]
  # print(distribution)

  # keep different bins the same amount of samples
  if len(bin_0_9) > 0:
    bin_0_9 = bin_0_9[np.random.choice(len(bin_0_9), len(bin_80_89))]
  if len(bin_10_19) > 0:
    bin_10_19 = bin_10_19[np.random.choice(len(bin_10_19), len(bin_80_89))]
  if len(bin_20_29) > 0:
    bin_20_29 = bin_20_29[np.random.choice(len(bin_20_29), len(bin_80_89))]
  if len(bin_30_39) > 0:
    bin_30_39 = bin_30_39[np.random.choice(len(bin_30_39), len(bin_80_89))]
  if len(bin_40_49) > 0:
    bin_40_49 = bin_40_49[np.random.choice(len(bin_40_49), len(bin_80_89))]
  if len(bin_50_59) > 0:
    bin_50_59 = bin_50_59[np.random.choice(len(bin_50_59), len(bin_80_89))]
  if len(bin_60_69) > 0:
    bin_60_69 = bin_60_69[np.random.choice(len(bin_60_69), len(bin_80_89))]
  if len(bin_70_79) > 0:
    bin_70_79 = bin_70_79[np.random.choice(len(bin_70_79), len(bin_80_89))]
  # bin_80_89 = bin_80_89[np.random.choice(len(bin_80_89), 10)]

  dist_norm_data = np.concatenate((bin_0_9, bin_10_19, bin_20_29, bin_30_39, bin_40_49,
                                  bin_50_59, bin_60_69, bin_70_79, bin_80_89, bin_90_100))

  # print("Distribution normalized data: ", dist_norm_data)
  # print("size of normalized data: ", len(dist_norm_data))
  file_name = 'normalized_' + os.path.basename(ground_truth_file)
  np.savez_compressed(os.path.join(os.path.dirname(ground_truth_file), file_name), dist_norm_data)
  

if __name__ == '__main__':
  # load config file
  config_filename = '../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # load the ground truth data
  ground_truth_file = config['overlap_file_path']
  normalize_data(ground_truth_file)

