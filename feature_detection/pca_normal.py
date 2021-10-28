# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # import ipdb;ipdb.set_trace()
    # 作业1
    # 屏蔽开始
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

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort] #note that eigenvectors[:,i] is ith eigenvector

    return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    # point_cloud_pynt = PyntCloud.from_file("/Users/renqian/Downloads/program/cloud_data/11.ply")
    # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # import ipdb;ipdb.set_trace()
    pcd = o3d.geometry.PointCloud()
    points = np.loadtxt("../../modelnet40_normal_resampled/airplane/airplane_0281.txt",delimiter=',')
    points = points[:,0:3]
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    # points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(pcd.points)
    point_cloud_vector = v[:, 2] #点云主方向对应的向量  ??? why the principal vector component is the last index?
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # o3d.visualization.draw_geometries([point_cloud_o3d])
    
    # 循环计算每个点的法向量
    # pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd) # create kd tree

    normals = []
    # import ipdb;ipdb.set_trace()
    # 作业2
    # 屏蔽开始
    for i in range(points.shape[0]):
        [k,idx,_] = pcd_tree.search_knn_vector_3d(pcd.points[i],200) # return idx:index of nearest points
        w,v = PCA(np.asarray(pcd.points)[idx[1:],:])  #input as numpy array, index starting from 1
        normals.append(v[:,-1])
        # normals.append(v[:,0])

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    print(f'current size of normals is: {len(normals)}')
    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
