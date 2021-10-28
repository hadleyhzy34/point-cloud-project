# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : iss.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/8/10 21:30
@Desc   : Intrinsic Shape Signatures Keypoint Detection
"""
import os
import random
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm


def parse_args():
    """Get command line arguments

    Returns:
        args (argparse.Namespace): arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat_idx", nargs="+", type=int, help="category index in modelnet40. 3 random category indices will be picked if this arg is not provided")
    parser.add_argument("--r", type=float, default=0.1, help="radius")
    parser.add_argument("--g21", type=float, default=0.8, help="gamma_21")
    parser.add_argument("--g32", type=float, default=0.8, help="gamma_32")
    return parser.parse_args()


def ISS(xyz, r, g21, g32):
    """Perform ISS keypoint detection on point cloud data

    Args:
        xyz (np.ndarray): point cloud data
        r (float): radius
        g21 (float): gamma21
        g32 (float): gamma32

    Returns:
        is_keypoint (list[bool]): a mask indicating whether point at any index is a keypoint or not
    """
    # Build kd-tree for point cloud data
    root = KDTree(xyz)
    # Initialize a numpy ndarray of length xyz.shape[0] and fill it with False
    # This is an indicator that at which index, point is treated as a keypoint or not
    is_keypoint = np.full(xyz.shape[0], False)
    # Initialize an empty list to store all the neighbors' indices that are within radius r of the current point
    ind_list = []
    # Initialize an empty list to store point's lambda_3 value
    l3_list = []

    print("Initializing keypoint proposal")
    # For each point (pi) in the point cloud
    for i, pi in tqdm(enumerate(xyz), total=xyz.shape[0]):
        # Perform radiusNN to get all neighbors' indices
        ind = root.query_ball_point(pi, r)
        # Store neighbors' indices
        ind_list.append(ind)
        # Get neighbor point set using their indices
        neighbors = xyz[ind]
        # Initialize a empty list to store weight of any neighbor point (wj)
        w = []
        # For each neighbor point in neighbor point set
        for neighbor in neighbors:
            # Append its weight (inverse of number of its neighbors within r)
            w.append(1 / len(root.query_ball_point(neighbor, r)))
        # Convert w to numpy ndarray
        w = np.asarray(w)
        # (pj - pi) in matrix format
        P = neighbors - pi
        # Compute Weighted covariance matrix Cov(pi)
        Cov = w * P.T @ P / np.sum(w)
        # Compute eigenvalues of Cov(pi) as lambda_1, lambda_2, lambda_3 in descending order
        e_values, e_vectors = np.linalg.eig(Cov)
        l1, l2, l3 = e_values[np.argsort(-e_values)]
        # Store point's lambda_3 value
        l3_list.append(l3)
        # Initialize keypoint proposal with the criterion: l2 / l1 < g21 and l3 / l2 < g32
        if l2 / l1 < g21 and l3 / l2 < g32:
            is_keypoint[i] = True

    print("Performing NMS")
    # For each point (pi) in the point cloud
    for i in tqdm(range(len(is_keypoint))):
        # Initialize an empty list to store keypoints' indices and lambda_3 values
        keypoint_list = []
        # If the point itself is a keypoint
        if is_keypoint[i]:
            # Store its index and lambda_3 value
            keypoint_list.append([i, l3_list[i]])
        # for each neighbor
        for j in ind_list[i]:
            # If the neighbor is itself, skip
            if j == i:
                continue
            # If the neighbor is a keypoint
            if is_keypoint[j]:
                # Store its index and lambda_3 value
                keypoint_list.append([j, l3_list[j]])
        # If there is no keypoints in keypoint_list, skip
        if len(keypoint_list) == 0:
            continue
        # Convert keypoint_list to numpy ndarray
        keypoint_list = np.asarray(keypoint_list)
        # Sort keypoint_list using lambda_3 value in descending order
        keypoint_list = keypoint_list[keypoint_list[:, -1].argsort()[::-1]]
        # Only the keypoint with the largest lambda_3 value is considered as the final keypoint
        # Get all the indices to be suppressed except for the first row
        filter_ind = keypoint_list[1:, 0].astype(int)
        # Set keypoint status of point at those indices to False
        is_keypoint[filter_ind] = False

    return is_keypoint


def load_data(args):
    """Load point cloud data of different categories into a list of numpy array

    Args:
        args (argparse.Namespace): arguments

    Returns:
        data (list[np.ndarray]): a list of point cloud data
        cat_names (list[str]): a list of category names
    """
    with open("../../data/modelnet40_normal_resampled/modelnet40_shape_names.txt") as f:
        shape_names = f.read().splitlines()
    if args.cat_idx:
        cat_names = [shape_names[i] for i in args.cat_idx]
    else:
        cat_names = random.sample(shape_names, 3)
    print(f"Loading point cloud data: {', '.join([cat_name + '_0001.txt' for cat_name in cat_names])}...")
    data_paths = [os.path.join("../../data/modelnet40_normal_resampled/", cat_name, f"{cat_name}_0001.txt") for cat_name in tqdm(cat_names)]
    data = [np.loadtxt(data_path, delimiter=",") for data_path in data_paths]
    return data, cat_names


def visualize_pcd_keypoint(keypoint_mask, xyz):
    """Visualize point cloud and its keypoints using open3d

    Args:
        keypoint_mask (np.ndarray): a numpy ndarray of boolean indicating each point's keypoint status
        xyz (np.ndarray): point cloud data
    """
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz))
    pcd_colors = np.tile([0.5, 0.5, 0.5], (xyz.shape[0], 1))
    pcd_colors[keypoint_mask] = np.array([1, 0, 0])
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    o3d.visualization.draw_geometries([pcd])


def main():
    # Get command line arguments
    args = parse_args()
    # Load point cloud data of different categories into a list of numpy array
    data, cat_names = load_data(args)
    for pcd_np, cat_name in zip(data, cat_names):
        print(f"Detecting keypoints in {cat_name + '_0001.txt'}...")
        xyz = pcd_np[:, :3]
        # Run ISS keypoint detection
        is_keypoints = ISS(xyz, r=args.r, g21=args.g21, g32=args.g32)
        # Visualize point cloud (gray) along with keypoints (red)
        visualize_pcd_keypoint(is_keypoints, xyz)


if __name__ == '__main__':
    main()
