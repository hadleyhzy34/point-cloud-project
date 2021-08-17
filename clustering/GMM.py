# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        # initialize prior probability distribution
        self.prior_dist = None
    
    # 屏蔽开始
    # 更新W
    

    # 更新pi
 
        
    # 更新Mu


    # 更新Var


    # 屏蔽结束
    
    def fit(self, data):
        data_min = np.min(data,axis=0)
        data_max = np.max(data,axis=0)

        num = data.shape[0]
        #initialize prior distribution probability
        self.prior_dist = np.ones((num,self.n_clusters))*(1/self.n_clusters)

        #initialize n_cluster guassian distribution
        mu = np.random.randn(self.k_,2)*(data_max-data_min)+data_min
        var = 
       
        for _ in range(self.max_iter_):
            # calculate prob for ith data and jth gaussian distribution

            # calculate pi

            # calculate mu

            # calculate var

            #

            #
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
    
    def predict(self, data):
        result = []
        distance = (np.array([data[:,0],]*self.k_).transpose()-self.centers[:,0])**2 + \
            (np.array([data[:,1],]*self.k_).transpose()-self.centers[:,1])**2
        
        for i in range(data.shape[0]):
            result.append(np.argmin(distance[i,:]))
        return result

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

