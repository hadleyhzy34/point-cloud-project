import os
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm
from visualize import read_bin_velodyne,kitti_random_pcl, pcl_random, pcl_random_visualize, pcl_data
from frnn.gpu_frnn import gpu_frnn
from nms.gpu_nms import gpu_nms

def iss(data, gamma21, gamma32, KDTree_radius, NMS_radius, max_num=100):
    """
    Description: intrinsic shape signatures algorithm
    Args:
        data: numpy array of point cloud, shape(num_points, 3)
        gamma21:
        gamma32:
        KDTree_radius: radiusNN range
        NMS_radius:
        max_num: max number of keypoints
    Return:
        is_keypoint->list[bool]: mask indicating whether point is a keypoint or not
    """
    # import ipdb;ipdb.set_trace()
    print(f'iss algo started...{data.shape[0]} of points prepared')

    #transfer data dtype to float32 before processing it on GPU
    temp_data = data.astype(np.float32)

    # create mask to indicate whether point i and point j are within range r
    adj = gpu_frnn(temp_data,KDTree_radius)
    adj = adj.reshape((data.shape[0],data.shape[0]))

    # initialize empty list to store neighbor points
    r_list = []
    l3_list = []
    is_keypoint = np.full(data.shape[0],False)

    # number of neighbor points for each point
    weights = np.sum(adj,axis=1)
    for i in tqdm(range(data.shape[0])):
        indices = np.argwhere(adj[i,:]>0)[:,0]
        weight = 1 / weights[indices]
        neighbors = data[indices]
        # store neighbor indices
        r_list.append(indices)
        # (pj - pi) in matrix format
        P = neighbors - data[i]
        # Compute Weighted covariance matrix Cov(pi)
        Cov = weight * P.T @ P / np.sum(weight)
        # Compute eigenvalues of Cov(pi) as lambda_1, lambda_2, lambda_3 in descending order
        e_values, e_vectors = np.linalg.eig(Cov)
        l1, l2, l3 = e_values[np.argsort(-e_values)]
        # Store point's lambda_3 value
        l3_list.append(l3)
        # Initialize keypoint proposal with the criterion: l2 / l1 < g21 and l3 / l2 < g32
        if l2 / l1 < gamma21 and l3 / l2 < gamma32:
            is_keypoint[i] = True

    print("performing nms based on cuda")
    l3_array = np.asarray(l3_list)
    gpu_nms(is_keypoint, l3_array, KDTree_radius)

    """
    # For each point (pi) in the point cloud
    for i in tqdm(range(len(is_keypoint))):
        # Initialize an empty list to store keypoints' indices and lambda_3 values
        keypoint_list = []
        # If the point itself is a keypoint
        if is_keypoint[i]:
            # Store its index and lambda_3 value
            keypoint_list.append([i, l3_list[i]])
        # for each neighbor
        for j in r_list[i]:
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
    """

    return is_keypoint

def main():
    """
    load pcl data
    calculate iss
    visualize iss points
    """
    pts = pcl_data()
    # pts = kitti_random_pcl()
    keypoint = iss(pts, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.3, max_num=5000)
    
    pts_colors = np.tile([0.5,0.5,0.5], (pts.shape[0], 1))
    pts_colors[keypoint] = np.array([1,0,0])
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(pts_colors)
    o3d.visualization.draw_geometries([pcd])
    """
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    key_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(keypoint))

    key_view.paint_uniform_color([1, 0, 0])
    pc_view.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries([key_view])
    o3d.visualization.draw_geometries([key_view, pc_view])
    """

if __name__ == "__main__":
    main()

