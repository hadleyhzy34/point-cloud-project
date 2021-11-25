import numpy as np
import open3d as o3d
import os

pcd = o3d.io.read_point_cloud("turtlebot3.ply")
print(pcd)
o3d.visualization.draw_geometries([pcd])
