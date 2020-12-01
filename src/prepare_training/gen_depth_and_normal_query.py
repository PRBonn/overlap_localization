#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates depth and normal data for query scans

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

import utils

try:
  from c_gen_depth_and_normal import gen_depth_and_normal
  from c_gen_virtual_scan import gen_virtual_scan
except:
  print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
  sys.exit(-1)


def gen_depth_and_normal_query(query_scan_paths, depth_folder, normal_folder, range_image_params):
  """ Generate depth and normal data for query scans.
    Args:
      query_scan_paths: paths of query scans.
      depth_folder: path of folder for generated depth data.
      normal_folder: path of folder for generated normal data.
      range_image_params: parameters for generating a range image.
  """
  if not os.path.exists(depth_folder):
    os.makedirs(depth_folder)
  
  if not os.path.exists(normal_folder):
    os.makedirs(normal_folder)
  
  print('start generating depth and normal data for query scans...')
  for query_scan_path in tqdm(query_scan_paths):
    # load virtual scan and the coordinate
    curren_points = utils.load_vertex(query_scan_path)
    frame_name = os.path.basename(query_scan_path).replace('.bin', '')
    
    # check existence
    if os.path.exists(os.path.join(depth_folder, frame_name + '.npy')):
      print('existing: ', frame_name)
      continue
    
    query_scan = gen_virtual_scan(curren_points.astype(np.float32),
                                  range_image_params['height'], range_image_params['width'],
                                  range_image_params['fov_up'], range_image_params['fov_down'],
                                  range_image_params['max_range'], range_image_params['min_range'])

    # generate depth and normal data
    depth_and_normal = gen_depth_and_normal(query_scan.astype(np.float32),
                                            range_image_params['height'], range_image_params['width'],
                                            range_image_params['fov_up'], range_image_params['fov_down'],
                                            range_image_params['max_range'], range_image_params['min_range'])

    depth = depth_and_normal[:, :, 3] / np.max(depth_and_normal[:, :, 3])
    normal = depth_and_normal[:, :, :3]
    
    # save depth and normal data
    np.save(os.path.join(depth_folder, frame_name), depth)
    np.save(os.path.join(normal_folder, frame_name), normal)


if __name__ == '__main__':
  # load config file
  config_filename = '../../config/prepare_training.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]

  if yaml.__version__>='5.1':  
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))

  depth_folder = '../' + config['query_depth_folder']
  normal_folder = '../' + config['query_normal_folder']

  # load virtual scans
  query_scan_folder = '../' + config['scan_folder']
  query_scan_paths = utils.load_files(query_scan_folder)
  
  range_image_params = config['range_image']

  gen_depth_and_normal_query(query_scan_paths[:100], depth_folder, normal_folder, range_image_params)

