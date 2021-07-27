import numpy as np
import open3d as o3d
import os


point_cloud = o3d.geometry.PointCloud()

pcd = np.loadtxt("../../modelnet40_normal_resampled/airplane/airplane_0281.txt",delimiter=',')

# pcd = np.loadtxt("../../modelnet40_normal_resampled/bathtub/bathtub_0002.txt",delimiter=',')

# pcd = o3d.io.read_point_cloud(pcd[:,0:3], format='xyz')
# o3d.visualization.draw_geometries([pcd])

# import ipdb;ipdb.set_trace()
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