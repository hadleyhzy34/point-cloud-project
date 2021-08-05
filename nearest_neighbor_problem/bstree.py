import numpy as np

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

# data generation
db_size = 10
data = np.random.permutation(db_size).tolist()
print(data)

data = np.array([8,3,10,1,6,4,7,14,13])


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

# generate bst by using insert method
root = None
for i,point in enumerate(data):
    root = insert(root, point, i)

print(f'------------------current inorder sequence of bst------------------------------')
inorder(root)
print(f'------------------current preorder sequence of bst------------------------------')
preorder(root)
print(f'------------------current postorder sequence of bst------------------------------')
postorder(root)

