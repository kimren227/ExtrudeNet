

import os
import glob
import random
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

CATEGORIES_IDS = {
    'airplane': '02691156',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'table': '04379243',
    'sofa':'04256520',
    'telephone': '04401088',
    'vessel':'04530566',
    'loudspeaker':'03691459',
    'cabinet': '02933112',
    'display':'03211117',
    'bench':'02828884',
    'rifle':'04090263'
    }


IDS_CATEGORIES = {index: category for category, index in CATEGORIES_IDS.items()}

def get_ids_from_category(*categories):
    return [CATEGORIES_IDS[category] for category in categories]

def get_category_from_ids(*indices):
    return [IDS_CATEGORIES[index] for index in indices]

class ShapeNet(Dataset):
    '''
    Lazy Loading ShapeNet Dataset with PointCloud and Occupancy
    '''
    def __init__(self, partition="train", categories=None, shapenet_root="./data/shapenet", implicite_function="Occupancy", balance=True, num_surface_points=2048, num_testing_points=2048):
        super().__init__()
        self.PARTITIONS = ["train", "test", "val"]
        self.shapenet_root = shapenet_root
        self.categories = categories
        self.partition = partition
        self.balance = balance
        self.implicite_function = implicite_function
        self.num_surface_points = num_surface_points
        self.num_testing_points = num_testing_points

        assert os.path.exists(self.shapenet_root), "Please download shapenet dataset and place it under 'data'"
        assert self.partition in self.PARTITIONS, "Partition must be either train, test or val.... "
        assert self.implicite_function in ["Occupancy", "SignedDisntaceFunction"], "Only Occupancy and SignedDisntaceFunction are supported"

        if categories == "all":
            categories = list(IDS_CATEGORIES.keys())
            print("Using all categories as training set...")
        else:
            print("Using categories: %s" % " ".join(self.categories))

        self.data_urls = []
        print("Reading %s splits of " % self.partition, end=" ")
        for category in categories:
            print("%s:%s" % (category, IDS_CATEGORIES[category]), end=" ")
            with open("%s/%s/%s.lst" % (self.shapenet_root, category, self.partition), "r") as f:
                self.data_urls += ["%s/%s/%s" % (self.shapenet_root, category, line.strip('\n')) for line in f.readlines()]
        print()

    def __getitem__(self, item):
        '''
        :param item: int
        :return: surface points [N, 3]
        :return: testing points with last bit indicating occupancy [M, 4]
        '''
        data_url = self.data_urls[item]

        # Loading Data file
        pointcloud = np.asarray(o3d.io.read_point_cloud(os.path.join(data_url, "model_surface_point_cloud.ply")).points)

        # Load testing points
        if self.implicite_function == "Occupancy":
            testing_points = np.load(os.path.join(data_url, "model_occupancy.npy"))
            # downsample testing point clouds
            if self.balance:
                inner_points = testing_points[testing_points[:,-1]==1]
                outer_points = testing_points[testing_points[:,-1]==0]
                inner_index = np.random.randint(0, inner_points.shape[0], self.num_testing_points//2)
                outer_index = np.random.randint(0, outer_points.shape[0], self.num_testing_points//2)
                testing_points = np.concatenate([inner_points[inner_index], outer_points[outer_index]], axis=0)
            else:
                testing_indices = np.random.randint(0, testing_points.shape[0], self.num_testing_points)
                testing_points = testing_points[testing_indices]
        else:
            testing_points = np.load(os.path.join(data_url, "model_sdf.npy"))
            # downsample testing point clouds
            if self.balance:
                inner_points = testing_points[testing_points[:,-1]<0]
                outer_points = testing_points[testing_points[:,-1]>=0]
                inner_index = np.random.randint(0, inner_points.shape[0], self.num_testing_points//2)
                outer_index = np.random.randint(0, outer_points.shape[0], self.num_testing_points//2)
                testing_points = np.concatenate([inner_points[inner_index], outer_points[outer_index]], axis=0)
            else:
                testing_indices = np.random.randint(0, testing_points.shape[0], self.num_testing_points)
                testing_points = testing_points[testing_indices]
        # downsample surface point clouds
        surface_indices = np.random.randint(0, pointcloud.shape[0], self.num_surface_points)
        pointcloud = pointcloud[surface_indices]
        return pointcloud.astype(np.float32), testing_points.astype(np.float32)

    def __len__(self):
        return len(self.data_urls)
