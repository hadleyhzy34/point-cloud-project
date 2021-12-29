import numpy as np
import random

STOP_THRESHOLD = 1e-4
CLUSTER_THRESHOLD = 1e-1

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)

class Dbscan:
    def __init__(self,distance,min_samples):
        self.eps = distance
        self.MinPts = min_samples
    
    def _exampndCluster(self, idx, neighborPts, C, unvisited, cluster_points, points):
        cluster_points[idx] = C # set core point to cluster C
        #unvisited.remove(idx)
        #import ipdb;ipdb.set_trace()
        while len(neighborPts) != 0:
            #print(f'current number of neighbor points is {len(neighborPts)}')
            #import ipdb;ipdb.set_trace()
            if neighborPts[0] in unvisited:
                unvisited.remove(neighborPts[0])
            neighborPts_target = self._regionQuery(neighborPts[0],points)
            if len(neighborPts_target) > self.MinPts:
                for n in neighborPts_target:
                    if n in unvisited:
                        neighborPts.append(n)
                #neighborPts = neighborPts + neighborPts_target
                neighborPts = list(dict.fromkeys(neighborPts))
            if cluster_points[neighborPts[0]] == 0:
                cluster_points[neighborPts[0]] = C
            del neighborPts[0]

    def dbscan(self, points):
        unvisited = list(range(0, points.shape[0]))
        cluster_points = np.zeros(points.shape[0],dtype=int)
        cur_cluster_id = 1

        while len(unvisited) > 0:
            i = unvisited[0]
            del unvisited[0]
            print(f'current length of unvisited is: {len(unvisited)}')
            neighborPts = self._regionQuery(i,points)
            #import ipdb;ipdb.set_trace()
            if len(neighborPts) > self.MinPts:
                #del unvisited[0]
                self._exampndCluster(i, 
                                     neighborPts,
                                     cur_cluster_id,
                                     unvisited,
                                     cluster_points,
                                     points)
                cur_cluster_id += 1
        return cluster_points

    def _regionQuery(self, idx, points):
        '''
        args:
            idx: query point
            points: points list
        return:
            neighborPts: index of neighborPts
        '''
        neighborPts = []
        for i,point in enumerate(points):
            if distance(points[idx], point) < self.eps:
                neighborPts.append(i)
        return neighborPts

#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 

def colors(n):
  ret = []
  for i in range(n):
    ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
  return ret

def main():
    centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.4)
    
    agent = Dbscan(0.4,15)
    dbscan_result = agent.dbscan(X)
    
    np.set_printoptions(precision=3)
    print('input: {}'.format(X))
    print('assined clusters: {}'.format(dbscan_result))
    color = colors(np.unique(dbscan_result).size)

    for i in range(len(dbscan_result)):
        plt.scatter(X[i, 0], X[i, 1], color = color[dbscan_result[i]])
    plt.show()

if __name__ == '__main__':
    main()
