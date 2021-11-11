import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

def PCA(data, correlation=False, sort=True):
    # import ipdb;ipdb.set_trace()
    # calculate mean value for each dimension/column
    M = np.mean(data,axis=0)
    # print(M.shape)

    # centered array
    pcd_center = data - M
    # print(pcd_center.shape)

    # calculate covariance matrix of centered matrix
    pcd_conv = np.cov(pcd_center.T)
    # print(pcd_conv)


    # eigen value, vector of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(pcd_conv)


    if sort:
        sort = eigenvalues.argsort()[::-1] # create a new sorted array in descending order
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort] #note that eigenvectors[:,i] is ith eigenvector

    return eigenvalues, eigenvectors


def main():
    # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # import ipdb;ipdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    points = np.loadtxt("../../modelnet40_normal_resampled/airplane/airplane_0281.txt",delimiter=',')
    points = points[:,0:3]
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([point_cloud_o3d])

    # points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    w, v = PCA(pcd.points)
    point_cloud_vector = v[:, 2] 
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # o3d.visualization.draw_geometries([point_cloud_o3d])
    
    # pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd) # create kd tree

    normals = []
    # import ipdb;ipdb.set_trace()
    for i in range(points.shape[0]):
        import ipdb;ipdb.set_trace()
        [k,idx,_] = pcd_tree.search_knn_vector_3d(pcd.points[i],200) # return idx:index of nearest points
        w,v = PCA(np.asarray(pcd.points)[idx[1:],:])  #input as numpy array, index starting from 1
        normals.append(v[:,-1])
        # normals.append(v[:,0])
    # print(f'current size of normals is: {len(normals)}')
    normals = np.array(normals, dtype=np.float64)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
