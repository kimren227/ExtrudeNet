from config import Config
import argparse
import open3d as o3d
import numpy as np
import os
import torch
import glob
from tqdm import tqdm
from dataset import ShapeNet
from chamfer_distance import ChamferDistance
from torch.utils.data import DataLoader

def chamfer_distance(source_cloud, target_cloud):
    source_cloud = torch.tensor(source_cloud).unsqueeze(0).cuda()
    target_cloud = torch.tensor(target_cloud).unsqueeze(0).cuda()
    chamferDist = ChamferDistance()
    distance_1, distance_2 = chamferDist(source_cloud, target_cloud)
    distance_1 = distance_1.mean()
    distance_2 = distance_2.mean()
    return distance_1.item() + distance_2.item()

def get_chamfer_distance(gt_pointcloud, output_mesh):
    mesh = o3d.io.read_triangle_mesh(output_mesh)
    pcd = mesh.sample_points_poisson_disk(20000)
    pred_points = np.asarray(pcd.points, dtype=np.float32)
    distance = chamfer_distance(gt_pointcloud, pred_points)
    return distance

def get_all_mesh_indices(mesh_dir):
    mesh_path = glob.glob(mesh_dir+"/*.ply")
    return [int(path.split("/")[-1].split(".")[0]) for path in mesh_path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ExtrudeNet')
    parser.add_argument('--config_path', type=str, default='./configs/plane.json', metavar='N',
                        help='config_path')
    args = parser.parse_args()
    config = Config((args.config_path))
    
    dataset = ShapeNet(shapenet_root=config.dataset_root, num_surface_points=20000, num_testing_points=2048, balance=False, categories=[config.category,], partition="val")
    samples_dir = os.path.join(config.sample_dir, config.experiment_name)
    dists = []

    for index in tqdm(get_all_mesh_indices(samples_dir)):
        try:
            pointcloud, gt_occupancies = dataset[index]
            pred_mesh = samples_dir+"/%d.ply" % index
            dist = get_chamfer_distance(pointcloud, pred_mesh)
            dists.append(dist)
        except:
            continue

    print("")
    print("=======%s========" % config.experiment_name)
    print("Scaled by 1000:")
    print(f"CD: {sum(dists)/len(dists)*1000}")
