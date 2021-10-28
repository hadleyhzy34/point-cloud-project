import numpy as np
import open3d as o3d
import os
import random
import struct

def pcl_random_visualize()->None:
    '''
    randomly select a pointcloud and visualize it
    '''
    # import ipdb;ipdb.set_trace()
    path = os.path.join(os.getcwd(),"../modelnet40_normal_resampled")
    category = ["airplane",
                "bathtub",
                "bed",
                "bench",
                "bookshelf",
                "bottle",
                "bowl",
                "car",
                "chair",
                "cone",
                "cup",
                "curtain",]
    cls = random.choice(category)
    path = os.path.join(path, cls)
    _, _, files = next(os.walk(path))
    file_name = str(random.randint(1,len(files)))
    sample = path + "/" + cls + "_" + "0" * (4-len(file_name)) + file_name + ".txt"
    pcd = np.loadtxt(sample, delimiter=',')
    # extract valuable dimensions
    pcd = pcd[:,0:3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([point_cloud])

def pcl_data():
    '''
    randomly select a pointcloud and return this numpy data
    '''
    # import ipdb;ipdb.set_trace()
    path = os.path.join(os.getcwd(),"../modelnet40_normal_resampled")
    category = ["airplane",
                "bathtub",
                "bed",
                "bench",
                "bookshelf",
                "bottle",
                "bowl",
                "car",
                "chair",
                "cone",
                "cup",
                "curtain",]
    cls = random.choice(category)
    path = os.path.join(path, cls)
    _, _, files = next(os.walk(path))
    file_name = str(random.randint(1,len(files)))
    sample = path + "/" + cls + "_" + "0" * (4-len(file_name)) + file_name + ".txt"
    pcd = np.loadtxt(sample, delimiter=',')
    # extract valuable dimensions
    return pcd[:,0:3]

def pcl_random():
    '''
    randomly select a pointcloud and return this object
    '''
    # import ipdb;ipdb.set_trace()
    path = os.path.join(os.getcwd(),"../modelnet40_normal_resampled")
    category = ["airplane",
                "bathtub",
                "bed",
                "bench",
                "bookshelf",
                "bottle",
                "bowl",
                "car",
                "chair",
                "cone",
                "cup",
                "curtain",]
    cls = random.choice(category)
    path = os.path.join(path, cls)
    _, _, files = next(os.walk(path))
    file_name = str(random.randint(1,len(files)))
    sample = path + "/" + cls + "_" + "0" * (4-len(file_name)) + file_name + ".txt"
    pcd = np.loadtxt(sample, delimiter=',')
    # extract valuable dimensions
    pcd = pcd[:,0:3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)
    # o3d.visualization.draw_geometries([point_cloud])
    return point_cloud

def read_bin_velodyne(path):
    # import ipdb;ipdb.set_trace()
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def kitti_random_visualize()->None:
    '''
    randomly select a pointcloud from KITTI dataset and visualize it
    '''
    # import ipdb;ipdb.set_trace()
    path = "/usr/data/pointbase/KITTI_DATASET_ROOT/training/velodyne"
    _, _, files = next(os.walk(path))
    sample_name = "00"+str(random.randint(0,len(files)-1))
    sample = path + "/" + sample_name + ".bin"
    pcd =read_bin_velodyne(sample)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([point_cloud])

def kitti_random_pcl()->None:
    '''
    randomly numpy type of point cloud from KITTI dataset and visualize it
    '''
    # import ipdb;ipdb.set_trace()
    path = "/usr/data/pointbase/KITTI_DATASET_ROOT/training/velodyne"
    _, _, files = next(os.walk(path))
    sample_name = "00"+str(random.randint(0,len(files)-1))
    sample = path + "/" + sample_name + ".bin"
    return read_bin_velodyne(sample)

if __name__ == "__main__":
    #pcl_random_visualize()
    kitti_random_visualize()
'''
path = os.path.join(os.getcwd(),"../modelnet40_normal_resampled")
pcd = np.loadtxt(path + "/airplane/airplane_0281.txt",delimiter=',')

# extract valuable dimensions
pcd = pcd[:,0:3]

# calculate mean value for each dimension/column
M = np.mean(pcd,axis=0)
print(M.shape)

# centered array
pcd_center = pcd - M
print(pcd_center.shape)

# calculate covariance matrix of centered matrix
pcd_conv = np.cov(pcd_center.T)
print(pcd_conv)


# eigen value, vector of covariance matrix
eigenValues, eigenVector = np.linalg.eig(pcd_conv)

print(eigenValues)

print(eigenVector)

# sort eigen value
sorted_index = np.argsort(eigenValues)[::-1]
print(f'last sorted index is: {sorted_index}')

# sorted eigen values and vectors
sorted_values = eigenValues[sorted_index]
sorted_vectors = eigenVector[:,sorted_index] # the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

print(f'sorted value is: {sorted_values}')
print(f'sorted vectors is: {sorted_vectors}')
point_cloud.points = o3d.utility.Vector3dVector(pcd[:,0:3])
point_cloud.points.append(sorted_vectors[0,:])
# print(pcd.shape)
# print(type(pcd))
# print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([point_cloud])
'''
