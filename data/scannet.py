# Ref:
# https://github.com/PointCloudYC/tensorflow/blob/main/main_S3DIS.py
# https://github.com/aRI0U/RandLA-Net-pytorch/blob/master/data.py

import os
import sys

sys.path.insert(0, '..')

import time
import glob
import pickle
import numpy as np
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader

from util.helper_ply import read_ply

try:
    from data.data_process import DataProcessing as DP
except:
    from data_process import DataProcessing as DP

import pdb


class SCANNET(Dataset):
    def __init__(self, path, sub_grid_size, weak_label_ratio):
        self.path = path
        self.sub_grid_size = sub_grid_size
        self.weak_label_ratio = weak_label_ratio

        with open(os.path.join(self.path, 'list', 'scannetv2_val.txt')) as f:
            self.val_split = f.read().splitlines()
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))  # scan the folder for all ply files

        # validation projection indice list, each item represents projection nearest id over a sub_pc for each corresponding raw pc pt
        self.val_proj = []

        # validation labels list, each item represent a validation sub_pc's label
        self.val_labels = []

        # possibility for control to randomly choose a point in the sub_pc evenly
        self.possibility = {}
        self.min_possibility = {}

        # {training, validation} sub_pc's kd_trees, colors, labels and names, and weak_label_mask
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_weak_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}

        # fill the above containers by reading physical sub_pc files
        self.load_sub_sampled_clouds(self.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        """load sub_sampled physical files and fill all the containers, input_{trees, colors, labels, names} and
        val_{proj, labels} and weak label_masks

        Args:
            sub_grid_size ([type]): sub-sampling grid size, e.g., 0.040
        """
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if cloud_name in self.val_split:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path,
                                '{:s}_KDTree.pkl'.format(cloud_name))  # e.g., Area_1_conferenceRoom_1_KDTree.pkl
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))  # e.g., Area_1_conferenceRoom_1.ply

            data = read_ply(sub_ply_file)  # ply format: x,y,z,red,gree,blue,class
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T  # (N',3), note the transpose symbol
            sub_labels = data['class']

            # read weak labels for sub_pc
            weak_label_folder = join(self.path, 'weak_label_{}'.format(self.weak_label_ratio))
            weak_label_sub_file = join(weak_label_folder, file_path.split('/')[-1][:-4] + '_sub_weak_label.ply')
            if os.path.exists(weak_label_sub_file):
                weak_data = read_ply(weak_label_sub_file)  # ply format: x,y,z,red,gree,blue,class
                weak_label_sub_mask = weak_data['weak_mask']  # (N',) to align same shape as sub_labels, 1 for labeled
            else:
                raise NotImplementedError(
                    "run the data_prepare_scannet.py to generate weak labels for raw and sub PC")

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # input_xx is a dict contain training or validation info for all sub_pc, each of them contain a list
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            # HACK: for validation set, all points should have labels meaning the weak_label_ratio is 1 (i.e., all points have labels)
            if cloud_split == 'validation':
                self.input_weak_labels[cloud_split] += [np.ones_like(weak_label_sub_mask)]
            else:
                self.input_weak_labels[cloud_split] += [weak_label_sub_mask]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')
        # Get validation and test reprojected indices and labels (this is useful for validating on all raw points)
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if cloud_name in self.val_split:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.all_files)


class IterableSCANNET(Dataset):
    def __init__(self, dataset, num_points=40960, loop=10, noise_init=3.5, split='training'):
        self.dataset = dataset
        self.num_points = num_points
        self.loop = loop
        self.noise_init = noise_init
        self.split = split

        self.possibility = {}
        self.min_possibility = {}

        # assign a possibility for all sub_pc and their containing points
        self.possibility[split] = []  # len = number of rooms in training/validation set
        self.min_possibility[split] = []  # same as above

        # random initialize
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __getitem__(self, item):
        return self.spatially_regular_gen()

    # def __iter__(self):
    #     return self.spatially_regular_gen()

    def __len__(self):
        return len(self.dataset.input_trees[self.split]) * self.loop

    def spatially_regular_gen(self):
        # choose the cloud with the lowest probability
        cloud_idx = int(np.argmin(self.min_possibility[self.split]))

        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.split][cloud_idx])

        # get all points within the cloud from tree structure
        points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)  # (N, 3)

        # center point of input region
        center_point = points[point_ind, :].reshape(1, -1)  # (1, 3)

        # add noise to the center point
        noise = np.random.normal(scale=self.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < self.num_points:
            # query all points within the cloud
            queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # query the predefined number of points
            queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=self.num_points)[1][0]

        # shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)

        # get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]  # (self.num_points, 3)
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
        queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]
        queried_pc_weak_label_mask = self.dataset.input_weak_labels[self.split][cloud_idx][queried_idx]

        # update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.split][cloud_idx][queried_idx] += delta
        self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

        # up_sampled with replacement
        if len(points) < self.num_points:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels, queried_pc_weak_label_mask = \
                DP.data_aug_weak(queried_pc_xyz, queried_pc_colors, queried_pc_labels,
                                queried_pc_weak_label_mask, queried_idx, self.num_points)

        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
        queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()  # (N,)
        queried_idx = torch.from_numpy(queried_idx).long()
        cloud_idx = torch.from_numpy(np.array([cloud_idx])).long()

        # yield (queried_pc_xyz, queried_pc_colors, queried_pc_labels,
        #        queried_pc_weak_label_mask, queried_idx, cloud_idx)
        return (queried_pc_xyz, queried_pc_colors, queried_pc_labels,
                queried_pc_weak_label_mask, queried_idx, cloud_idx)


if __name__ == '__main__':

    path = '../dataset/scannet'
    sub_grid_size = 0.04
    weak_label_ratio = 0.001
    dataset = SCANNET(path, sub_grid_size, weak_label_ratio)

    num_points = 40960
    loop = 10
    noise_init = 3.5

    batch_size = 16
    # train_steps = 2000  # // batch_size    # iterate over train_steps samples per training epoch
    split = 'training'
    train_dataset = IterableSCANNET(dataset, num_points, loop=loop, noise_init=noise_init, split=split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=True, sampler=None, drop_last=True)
    print('Train length: ', train_dataset.__len__())

    batch_size = 4
    # valid_steps = 400  # // batch_size     # iterate over valid_steps samples per validation epoch
    split = 'validation'
    valid_dataset = IterableSCANNET(dataset, num_points, loop=loop, noise_init=noise_init, split=split)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                              pin_memory=True, sampler=None, drop_last=False)
    print('Valid length: ', valid_dataset.__len__())

    for i, data in enumerate(train_loader):
        # xyz, colors, labels, mask, idx, cloud_idx = data

        print('{}/{}'.format(i + 1, len(train_loader)))

    for i, data in enumerate(valid_loader):
        # xyz, colors, labels, mask, idx, cloud_idx = data

        print('{}/{}'.format(i + 1, len(valid_loader)))
