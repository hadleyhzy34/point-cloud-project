# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    points = np.asarray(point_cloud) #change pcd.points data type to numpy array

    # import ipdb;ipdb.set_trace()
    # 作业3
    # 屏蔽开始
    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)

    d_x = (points_max[0] - points_min[0])//leaf_size #note to use // instead of /
    d_y = (points_max[1] - points_min[1])//leaf_size
    d_z = (points_max[2] - points_min[2])//leaf_size

    idx = np.zeros(points.shape[0])

    for i in range(points.shape[0]):
        # if i == 9999:
        #     import ipdb;ipdb.set_trace()
        #     print(i)
        # print(f'current index is: {i}')
        h_x = (points[i,0]-points_min[0])//leaf_size
        h_y = (points[i,1]-points_min[1])//leaf_size
        h_z = (points[i,2]-points_min[2])//leaf_size
        idx[i] = h_x+h_y*d_x+h_z*d_y*d_z
    
    sort_idx = idx.argsort()[::-1]  # be careful to add () after argsort

    # # centroid method
    # centroid = []
    # sum = np.zeros(3)
    # point_count = 0
    # for i in range(points.shape[0]):
    #     if i!=0 and idx[sort_idx[i]]!=idx[sort_idx[i-1]]:
    #         centroid.append(sum/point_count)
    #         sum = points[sort_idx[i],:]
    #         point_count = 1
    #     else:
    #         sum = sum + points[sort_idx[i],:]
    #         point_count += 1
        
    #     if i == points.shape[0]-1:
    #         centroid.append(sum/point_count)

    # random method
    centroid = []
    temp = []
    for i in range(points.shape[0]):
        # print(i)
        # if i == 9999:
        #     import ipdb;ipdb.set_trace()
        if i!=0 and idx[sort_idx[i]]!=idx[sort_idx[i-1]]:
            centroid.append(temp[int(np.random.rand()*len(temp))])
            temp = [points[sort_idx[i],:]]
        else:
            temp.append(points[sort_idx[i]])
        
        if i == points.shape[0]-1:
            centroid.append(temp[int(np.random.rand()*len(temp))])
    
    filtered_points = centroid
    # 屏蔽结束
    
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    # file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    # point_cloud_pynt = PyntCloud.from_file(file_name)
    point_cloud_o3d = o3d.geometry.PointCloud()
    points = np.loadtxt("../../modelnet40_normal_resampled/airplane/airplane_0281.txt",delimiter=',')
    points = points[:,0:3]
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)

    # 转成open3d能识别的格式
    # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_o3d.points, 0.1)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
