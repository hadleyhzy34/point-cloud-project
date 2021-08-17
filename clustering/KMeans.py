# 文件功能： 实现 K-Means 算法

import numpy as np
import sys
from matplotlib import colors, pyplot as plt
plt.style.use('seaborn')

from numpy.core.defchararray import center

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers = np.zeros((self.k_,3)) #self.centers: center_x, center_y, number of points belong to

    def fit(self, data):
        # initialize centers
        data_min = np.min(data,axis=0)
        data_max = np.max(data,axis=0)
        self.centers[:,0:2] = np.random.randn(self.k_,2)*(data_max-data_min)+data_min

        distance = np.zeros((data.shape[0],self.k_))
        for _ in range(self.max_iter_):
            # for i in range(data.shape[0]):
            #     for j in range(self.k_):
            #         distance[i][j] = (data[i][0]-self.centers[j][0])**2+(data[i][1]-self.centers[j][1])**2
            distance = (np.array([data[:,0],]*self.k_).transpose()-self.centers[:,0])**2 + \
            (np.array([data[:,1],]*self.k_).transpose()-self.centers[:,1])**2
            # recalculate centers
            self.centers = np.zeros((self.k_,3))
            for i in range(data.shape[0]):
                index = np.argmin(distance[i,:])
                self.centers[index] += np.array([data[i][0],data[i][1],1])
            
            for j in range(self.k_):
                if self.centers[j][2] == 0:
                    continue
                self.centers[j,0:2] = self.centers[j,0:2]/self.centers[j][2]
                # print(f'current centers are: {self.centers[j]}')

    def predict(self, p_datas):
        result = []
        distance = (np.array([p_datas[:,0],]*self.k_).transpose()-self.centers[:,0])**2 + \
            (np.array([p_datas[:,1],]*self.k_).transpose()-self.centers[:,1])**2
        
        for i in range(p_datas.shape[0]):
            result.append(np.argmin(distance[i,:]))
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],[100,100],[50,50]])
    k_means = K_Means(n_clusters=3)
    k_means.fit(x)

    plt.scatter(x[:,0], x[:,1],c='b')
    plt.scatter(k_means.centers[:,0],k_means.centers[:,1],c='g')
    plt.show()

    cat = k_means.predict(x)
    print(cat)

