"""
prepare S3DIS dataset, reproduced based on the paper (https://arxiv.org/abs/2104.04891)
Author: Chao YIN
Email: cyinac@connect.ust.hk

history: 
- Oct. 15, 2021, init the file
- Oct. 26, 2021, **fix a fatal bug which is primarily caused by misinterpretation of weak label ration.**
codebase: data_prepare_s3dis.py of the official RandLA-Net

difference from the codebase (data_prepare_s3dis.py of Official RandLA-Net) 
- add CLI arguments (e.g., sub_grid_size, weak_label_ratio) support with argparse
- generate separate weak labels for each room in S3DIS
- refactor the code; if the raw/sub-pc/kdtree/projected indices/weak_labels files exist, then read them into memory. 
"""

import os
import sys
import glob
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from helper_ply import write_ply, read_ply
from helper_tool import DataProcessing as DP

NYUCLASSES = [
    'floor',
    'wall',
    'cabinet',
    'bed',
    'chair',
    'sofa',
    'table',
    'door',
    'window',
    'bookshelf',
    'picture',
    'counter',
    'desk',
    'curtain',
    'refrigerator',
    'bathtub',
    'shower curtain',
    'toilet',
    'sink',
    'otherprop'
]


def convert_pc2plyandweaklabels(anno_path, save_path, sub_pc_folder, weak_label_folder,
                                weak_label_ratio, sub_grid_size, gt_class, gt_class2label):
    """
    Convert original dataset files (consiting of rooms) to ply file and weak labels.
    Physically, each room will generate several files,
    including raw_pc.ply, sub_pc.ply, sub_pc.pkl (for the kdtree), proj_idx.pkl (for each raw point's nearest neighbor in the sub_pc) and weak labels for raw and sub_pc, respectively
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
                                           e.g. scene0000_00.npy
    :param save_path: path to save original point clouds (each line is XYZRGBL), e.g., xx.ply
    :param sub_pc_folder: path to save subsampled point clouds
    :param weak_label_folder: path to save weak labeled point clouds
    :param weak_label_ratio: e.g., 0.001
    :param sub_grid_size: voxel size for sub-sample use
    :param gt_class
    :param gt_class2label
    :return: None
    """

    # num_raw_points = 0  # number of raw points for current room
    # num_sub_points = 0  # number of sub-sampled points for current room

    # save raw_cloud
    if not os.path.exists(save_path):
        # translate the data by xyz_min--yc
        pc_label = np.load(anno_path) # xyz rgb normal instancelabel semantic label
        xyz_min = np.amin(pc_label, axis=0)[0:3]
        pc_label[:, 0:3] -= xyz_min
        # manage data types and save in PLY format--yc
        xyz = pc_label[:, :3].astype(np.float32)
        num_raw_points = xyz.shape[0]
        colors = pc_label[:, 3:6].astype(np.uint8)

        # labels = pc_label[:, 10].astype(np.uint8)
        # write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        write_ply(save_path, (xyz, colors), ['x', 'y', 'z', 'red', 'green', 'blue'])
    else:
        # if existed then read this ply file to fill the data/xyz/colors/labels
        data = read_ply(save_path)  # ply format: x, y, z, red, green, blue, class
        xyz = np.vstack((data['x'], data['y'], data['z'])).T  # (N',3), note the transpose symbol
        num_raw_points = xyz.shape[0]
        colors = np.vstack((data['red'], data['green'], data['blue'])).T  # (N',3), note the transpose symbol
        # labels = data['class']
        # pc_label = np.concatenate((xyz, colors, np.expand_dims(labels, axis=1)), axis=1)
    print('here?')
    # save sub_cloud
    sub_ply_file = os.path.join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    print(save_path)
    print(sub_ply_file)
    if not os.path.exists(sub_ply_file):
        sub_xyz, sub_colors = DP.grid_sub_sampling(xyz, colors, grid_size=sub_grid_size)
        # sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
        sub_colors = sub_colors / 255.0
        num_sub_points = sub_xyz.shape[0]
        write_ply(sub_ply_file, [sub_xyz, sub_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    else:
        data = read_ply(sub_ply_file)  # ply format: x,y,z,red,gree,blue,class
        sub_xyz = np.vstack((data['x'], data['y'], data['z'])).T  # (N',3), note the transpose symbol
        num_sub_points = sub_xyz.shape[0]
        sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T  # (N',3), note the transpose symbol
        sub_labels = data['class']
    print('here?')
    # save KDTree for sub_pc
    kd_tree_file = os.path.join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
    search_tree = None
    if not os.path.exists(kd_tree_file):
        search_tree = KDTree(sub_xyz)
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)
    else:
        with open(kd_tree_file, 'rb') as f:
            search_tree = pickle.load(f)

    # save projection indcies for all raw points over the corresponding sub_pc
    proj_save = os.path.join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')
    if not os.path.exists(proj_save):
        proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        with open(proj_save, 'wb') as f:
            # pickle.dump([proj_idx, labels], f)
            pickle.dump([proj_idx], f)

    # USED for weakly semantic segmentation, save sub pc's weak labels
    # KEY: Randomly select some points to own labels, give them a mask (no need to save weak label mask for raw pc)
    weak_label_sub_file = os.path.join(weak_label_folder, save_path.split('/')[-1][:-4] + '_sub_weak_label.ply')
    if not os.path.exists(weak_label_sub_file):

        # compute weak ratio of weak points w.r.t. #sub_pc
        print(f'Current sub-sampled ratio(#sub_points/#raw_points) is {(num_sub_points / num_raw_points) * 100:.2f}%')
        print(f'Current weak_ratio(#weak_points/#raw_points) is {weak_label_ratio:.4f}')

        # set weak points by randomly selecting weak_label_ratio*N points(i.e., the number of raw_pc) and denote them w. a mask
        weak_label_sub_mask = np.zeros((num_sub_points, 1), dtype=np.uint8)

        # BUG FIXED: fixed already; here, should set replace = True, otherwise a bug will be resulted
        # KEY: weak_label_ratio should be multiplied by number of raw points rather sub-sampled points
        selected_idx = np.random.choice(num_sub_points, int(num_raw_points * weak_label_ratio), replace=False)
        weak_label_sub_mask[selected_idx, :] = 1
        write_ply(weak_label_sub_file, (weak_label_sub_mask,), ['weak_mask'])
    else:
        data = read_ply(weak_label_sub_file)
        weak_label_mask = data['weak_mask']


"""
Prepare the S3DIS dataset for training by generating new info from each room's point cloud(PC)
- input: each room's PC
- output: 
  - 1) raw_pc in ply, 
  - 2) sub_pc in ply, 
  - 3) KDTree for the sub_pc,
  - 4) projection indices for each raw point over the sub_pc(used for DL validation/inference as the learning process only occur on the sub_pc, therefore by relating raw point's relations to the sub_pc can help propagate their semantics.)
  - 5) weak labels for raw and sub_pc
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rng_seed', type=int, default=123, help='manual seed')
    parser.add_argument('--dataset_path', type=str, default='./data/SCANNET_TEST/test_fullarea',
                        help='dataset path')
    parser.add_argument('--sub_grid_size', type=float, default=0.04, help='grid-sampling size')
    parser.add_argument('--weak_label_ratio', type=float, default=0.001, help='the weakly semantic segmentation ratio')
    parser.add_argument('--out_format', type=str, default='.ply', help='output format, e.g., ply')
    FLAGS = parser.parse_args()

    # set fixed seeds for reproducible results
    random.seed(FLAGS.rng_seed)
    np.random.seed(FLAGS.rng_seed)

    dataset_path = FLAGS.dataset_path
    anno_paths = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path)]
    # object categories for the SCANNET dataset
    gt_class = NYUCLASSES
    gt_class2label = {cls: i for i, cls in enumerate(gt_class)}
    sub_grid_size = FLAGS.sub_grid_size  # grid_subsampling size

    """
    create 3 folder
    - input_0.040, for sub_pc.py, kdtree for sub_pc and the projection indices
    - original_ply, raw_pc.ply
    - weak_label_0.01, weak labels for raw and sub_pc
    """
    original_pc_folder = os.path.join(os.path.dirname(dataset_path), 'original_ply')
    sub_pc_folder = os.path.join(os.path.dirname(dataset_path), 'input_{:.3f}'.format(sub_grid_size))
    weak_label_ratio = FLAGS.weak_label_ratio
    weak_label_folder = os.path.join(os.path.dirname(dataset_path), 'weak_label_{}'.format(weak_label_ratio))
    os.mkdir(original_pc_folder) if not os.path.exists(original_pc_folder) else None
    os.mkdir(sub_pc_folder) if not os.path.exists(sub_pc_folder) else None
    os.mkdir(weak_label_folder) if not os.path.exists(weak_label_folder) else None
    out_format = FLAGS.out_format

    for annotation_path in anno_paths:
        # e.g.: data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/Annotations
        print(annotation_path)
        elements = str(annotation_path).split('/')
        # e.g.: Area_1_conferenceRoom_1.ply
        out_file_name = elements[-1].split('.')[0] + out_format

        # convert each room's pc to ply and more(kdtree and projection indices for raw points over its corresponding sub_pc)
        convert_pc2plyandweaklabels(annotation_path, os.path.join(original_pc_folder, out_file_name),
                                    sub_pc_folder, weak_label_folder,
                                    weak_label_ratio, sub_grid_size,
                                    gt_class, gt_class2label)
