import os
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm
from visualize import read_bin_velodyne,kitti_random_pcl, pcl_random, pcl_random_visualize, pcl_data

def iss(data, gamma21, gamma32, KDTree_radius, NMS_radius, max_num=100):
    """
    Description: intrinsic shape signatures algorithm
    Args:
        data: numpy array of point cloud
        gamma21:
        gamma32:
        KDTree_radius: leaf_size
        NMS_radius:
        max_num:
    Return:
    """
    import ipdb;ipdb.set_trace()
    print(f'iss algo started...{data.shape[0]} of points prepared')

    # KD tree to find nearest points within radius r
    leaf_size = 32
    tree = KDTree(data, leaf_size)
    # tree.query_ball_point(): Find all points within distance r of point(s) x.
    # If x is a single point, returns a list of the indices of the neighbors of x.
    # If x is an array of points, returns an object array of shape tuple
    # containing lists of neighbors.
    radius_neighbor = tree.query_ball_point(data, KDTree_radius)

    print("-" * 10, "start to search keypoints", '-' * 10)
    keypoints = []  # 首先定义keypoints集合
    min_feature_value = []  # 定义最小特征值集合 TODO 何用
    for index in tqdm(range(len(radius_neighbor))):
        neighbor_idx = radius_neighbor[index]   # 得到第一个点的邻近点索引
        neighbor_idx.remove(index)  # 把自己去掉
        # 如果一个点没有邻近点，直接跳过 TODO 邻近点少的是否也可以直接跳过不要了
        if len(neighbor_idx)==0:
            continue

        # 计算权重矩阵
        import ipdb;ipdb.set_trace()
        weight = np.linalg.norm(data[neighbor_idx] - data[index], axis=1)
        weight[weight == 0] = 0.001  # 避免除0的情况出现
        weight = 1 / weight

        # 直接循环求解，计算加权协方差矩阵
        cov = np.zeros((3, 3))
        # 这里需要加个np.newaxis，变成N*3*1，这样tmp[i]才能是3*1的矩阵
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]
        for i in range(len(neighbor_idx)):
            cov += weight[i]*tmp[i].dot(tmp[i].transpose())
        cov /= np.sum(weight)

        # 或者用下边的方法计算加权协方差矩阵
        '''
        tmp = (data[neighbor_idx] - data[index])[:, :, np.newaxis]  # N,3,1
        cov = np.sum(weight[:, np.newaxis, np.newaxis] *
                     (tmp @ tmp.transpose(0, 2, 1)), axis=0) / np.sum(weight)
        '''

        # 做特征值分解，SVD
        s = np.linalg.svd(cov, compute_uv=False)    # 不必计算u和vh，只计算特征值即可，默认为True
        # 根据特征值判断是否为特征点
        if (s[1]/s[0] < gamma21) and (s[2]/s[1] < gamma32):
            keypoints.append(data[index])
            min_feature_value.append(s[2])
    print("search keypoints finished", keypoints.__len__(), "points")   # print可以这样使用

    # NMS step
    # 非极大值抑制（Non-Maximum Suppression，NMS），顾名思义就是抑制不是极大值的元素，可以理解为局部最大搜索。
    # 也就是找到局部中最像特征值的那个点
    print("-"*10, "NMS to filter keypoints", "-"*10)
    keypoints_after_NMS = []
    leaf_size = 10  # 又来了一个leaf_size
    nms_tree = KDTree(keypoints, leaf_size)
    index_all = [i for i in range(len(keypoints))]
    for iter in tqdm(range(max_num)):
        # 找到s2特征值集合中最大点的索引
        max_index = min_feature_value.index(max(min_feature_value))
        # max_index = np.argmax(min_feature_value)
        tmp_point = keypoints[max_index]
        # 找到s2特征值集合中最大点的邻近
        del_indexs = nms_tree.query_ball_point(tmp_point, NMS_radius)
        # 删去找到的邻近，之后要只保留最大的s2特征值点对应的keypoint
        for del_index in del_indexs:
            if del_index in index_all:
                del min_feature_value[index_all.index(del_index)]   # 删去对应的特征值
                del keypoints[index_all.index(del_index)]           # 删去对应的关键点
                del index_all[index_all.index(del_index)]           # 删去对应的索引
        # NSM:保留最大的s2特征值点对应的keypoints
        keypoints_after_NMS.append(tmp_point)
        # 如果此时keypoints已经为0了，那就可以break了
        if len(keypoints) == 0:
            break
    print("NMS finished,find ", len(keypoints_after_NMS), " points")

    return np.array(keypoints_after_NMS)


def main():
    pts = pcl_data()
    keypoint = iss(pts, gamma21=0.6, gamma32=0.6, KDTree_radius=0.15, NMS_radius=0.3, max_num=5000)

    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    key_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(keypoint))

    key_view.paint_uniform_color([1, 0, 0])
    pc_view.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries([key_view])
    o3d.visualization.draw_geometries([key_view, pc_view])

if __name__ == "__main__":
    main()

