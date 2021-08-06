# kdtree的具体实现，包括构建和查找

import random
import math
import sys
from collections import deque
import numpy as np

class KNNContainer:
    '''
    container with deque structure to contain all candidate k nearest neighbor points
    '''
    def __init__(self, k):
        self.k = k
        self.worst_dist = sys.maxsize
        self.container = deque(maxlen=self.k) # deque container, [(distance, index)]
        for i in range(k):
            self.container.append((self.worst_dist,0))
    
    def add_point(self, dist, index):
        if dist > self.worst_dist: # if distance is already larger current kth nearest point
            return
        
        for i in range(len(self.container)):
            if dist < self.container[i][0]:
                self.container.pop() # pop right last neighbor point
                self.container.insert(i,(dist,index)) # insert ith point
                break
        
        self.worst_dist = self.container[self.k-1][0]

class RadiusContainer:
    '''
    container with deque structure to contain all candidate nearest neighbor points within radius range
    '''
    def __init__(self, radius):
        self.radius = radius
        self.container = deque() # deque container, [(distance, index)]
        # for i in range(k):
        #     self.container.append((self.worst_dist,0))
    
    def add_point(self, dist, index):
        if dist > self.radius: # if distance is already larger current kth nearest point
            return
        
        self.container.append((dist,index))


# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis  # splitting direction
        self.value = value # splitting point
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    # import ipdb;ipdb.set_trace()
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1

# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M
        
        middle_left_idx = math.ceil(point_indices_sorted.shape[0]/2)-1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]

        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        # median value
        root.value = (middle_left_point_value + middle_right_point_value) * 0.5
        # split direction
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis,dim=db.shape[1]),
                                           leaf_size)
        
        root.right = kdtree_recursive_build(root.right,
                                            db,
                                            point_indices_sorted[middle_right_idx:],
                                            axis_round_robin(axis,dim=db.shape[1]),
                                            leaf_size)
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNContainer, query: np.ndarray):
    '''
    func: find knn search
    input:
        root: kd tree
        db: initial data
        result_set: current ordered list points
    output:
        failure then output false
    '''
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_dist:
            return kdtree_knn_search(root.right, db, result_set,query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_dist:
            return kdtree_knn_search(root.left, db, result_set,query)

    return False


def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusContainer, query: np.ndarray):
    '''
    func:radius search
    input:
        root: kd tree
        db: initial data
        result_set: current ordered list points
    output:
        failure then output false
    '''
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.radius:
            return kdtree_radius_search(root.right, db, result_set,query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.radius:
            return kdtree_radius_search(root.left, db, result_set,query)
    # 屏蔽结束

    return False



def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 5

    db_np = np.random.rand(db_size, dim)
    print(db_np)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    query = np.asarray([0, 0, 0])
    result_set = KNNContainer(k)
    kdtree_knn_search(root, db_np, result_set, query)
    
    print(result_set.container)
    for c in result_set.container:
        print(db_np[c[1]])
    
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    print("Radius search:")
    query = np.asarray([0, 0, 0])
    result_set = RadiusContainer(radius = 0.5)
    kdtree_radius_search(root, db_np, result_set, query)
    
    print(result_set.container)
    for c in result_set.container:
        print(db_np[c[1]])


if __name__ == '__main__':
    main()