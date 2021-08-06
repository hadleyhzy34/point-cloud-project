import numpy as np
import sys
import math
from collections import deque

# Node Class
class Node:
    '''
    key: node.value
    value: index of node.value in list or array
    left: node.left
    right: node.right
    '''
    def __init__(self, key, value, left=None, right=None):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

def insert(root, key, value = -1):
    '''
    recursively construct bst
    '''
    if root is None:
        root = Node(key,value)
    else:
        if key < root.key:
            root.left = insert(root.left, key, value)
        elif key > root.key:
            root.right = insert(root.right, key, value)
        else:
            pass
    return root

def search_recursive(root, key):
    '''
    recursively search node
    '''
    if root is None or root.key == key:
        return root
    if key < root.key:
        return search_recursive(root.left, key)
    elif key > root.key:
        return search_recursive(root.right, key)


def search_iterative(root, key):
    '''
    iterative search node
    '''
    current_node = root
    while current_node is not None:
        if current_node.key == key:
            return current_node
        elif current_node.key < key:
            current_node = root.left
        elif current_node.key > key:
            current_node = root.right
    return current_node

def inorder(root):
    '''
    inorder search node, sorting
    '''
    if root is not None:
        inorder(root.left)
        print(root.key)
        inorder(root.right)

def preorder(root):
    '''
    preorder search node, copy tree
    '''
    if root is not None:
        print(root.key)
        preorder(root.left)
        preorder(root.right)

def postorder(root):
    '''
    post order search node, delete a node
    '''
    if root is not None:
        postorder(root.left)
        postorder(root.right)
        print(root.key)

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

def knn_search(root:Node, res:KNNContainer, key):
    '''
    return true when it finds k nearest points
    return false when it may not find all k nearest points
    '''
    if root is None:
        return False
    res.add_point(math.fabs(root.key-key),root.value)

    if res.worst_dist == 0:
        return True
    
    # if root.key >= key:
    #     if knn_search(root.left, res, key):
    #         return True
    #     elif math.fabs(root.key-key) < res.worst_dist:
    #         return knn_search(root.right,res,key)
    #     return False
    # else:
    #     if knn_search(root.right, res, key):
    #         return True
    #     elif math.fabs(root.key-key) < res.worst_dist:
    #         return knn_search(root.left, res, key)
    #     return False

    if root.key >= key:
        knn_search(root.left, res, key)
        if math.fabs(root.key-key) < res.worst_dist:
            return knn_search(root.right,res,key)
        return False
    else:
        knn_search(root.right, res, key)
        if math.fabs(root.key-key) < res.worst_dist:
            return knn_search(root.left, res, key)
        return False
    

# data generation
db_size = 500
data = np.random.permutation(db_size).tolist()
print(data)

# data = np.array([8,3,10,1,6,4,7,14,13])
        

# generate bst by using insert method
root = None
for i,point in enumerate(data):
    root = insert(root, point, i)

# print(f'------------------current inorder sequence of bst------------------------------')
# inorder(root)
# print(f'------------------current preorder sequence of bst------------------------------')
# preorder(root)
# print(f'------------------current postorder sequence of bst------------------------------')
# postorder(root)


k = 5
query_key = 101
res = KNNContainer(k)
knn_search(root, res, query_key)
print(res.container)
for c in res.container:
    print(data[c[1]])