# GMM algorithm

# to do list: 
# 1.matrix calculation for matrix inside fit function
# 2.draw trajetory for mu update history
# 3.make it stopped when convergence
# 4.prevent cov too small or too large

import numpy as np
from numpy import *
import numpy
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
        # import ipdb;ipdb.set_trace()
        # initialize parameters
        data_min = np.min(data,axis=0)
        data_max = np.max(data,axis=0)

        num = data.shape[0] # number of data
        #initialize prior distribution probability
        self.prior_dist = np.ones(self.n_clusters)*(1/self.n_clusters) # pi, (n_clusters)

        #initialize n_cluster guassian distribution
        self.mu = np.random.randn(self.n_clusters,2)*(data_max-data_min)+data_min  # mu, (n_clusters, 2)
        # self.var = np.ones((self.n_clusters,2,2)) * np.cov(data.T) # var, (n_clusters, 2, 2)
        self.var = np.ones((self.n_clusters,2,2)) * (np.eye(2) * np.array([np.cov(data.T)[0][0],np.cov(data.T)[1][1]]))
        # self.var = np.cov(data.T)  # var, (2,2)

        # initialize posterior probability
        self.epsilon = np.zeros((num,self.n_clusters)) # (num, n_clusters)

        # initialize multivariate guassian probability density function
        self.distribution = np.ones((num,self.n_clusters)) # (num, n_clusters)
        
        for i in range(num):
            for j in range(self.n_clusters):
                self.distribution[i][j] = multivariate_normal.pdf(data[i], mean=self.mu[j],cov=self.var[j])
       
        # import ipdb;ipdb.set_trace()
        for _ in range(self.max_iter):
            # # calculate distance from points to gausian mu
            # distance = (np.array([data[:,0],]*self.k_).transpose()-self.centers[:,0])**2 + \
            #     (np.array([data[:,1],]*self.k_).transpose()-self.centers[:,1])**2
            
            # calculate posterior probability
            self.epsilon = self.prior_dist*self.distribution / (np.array([np.mean(self.prior_dist*self.distribution,axis=1),]*self.n_clusters).transpose())
            
            # update mu
            self.mu = np.zeros((self.n_clusters,2))
            for j in range(self.n_clusters):
                self.mu[j] = np.array(np.sum(data[:,0]*self.epsilon[:,j]),np.sum(data[:,1]*self.epsilon[:,j]))
            # for j in range(self.n_clusters):
            #     for i in range(num):
            #         self.mu[j] += data[i] * self.epsilon[i,j]
            #     # import ipdb;ipdb.set_trace()
            #     self.mu[j] = self.mu[j] / np.sum(self.epsilon[:,j])

            # import ipdb;ipdb.set_trace()
            # calculate pi
            self.prior_dist = np.sum(self.epsilon,axis=0)/self.n_clusters

            # import ipdb;ipdb.set_trace()
            for j in range(self.n_clusters):
                for i in range(num):
                    self.var[j] += self.epsilon[i,j]*(np.eye(2)*np.diag(np.dot(np.array([data[i]-self.mu[j],]).transpose(),np.array([data[i]-self.mu[j],]))))
                self.var[j] = self.var[j] / np.sum(self.epsilon[:,j])

            # update distribution
            for i in range(num):
                for j in range(self.n_clusters):
                    # print(f'current data: {data[i]}, mean: {self.mu[j]}, var: {self.var[j]}')
                    self.distribution[i][j] = multivariate_normal.pdf(data[i], mean=self.mu[j],cov=self.var[j])

    def predict(self, data):
        result_x = [[] for _ in range(self.n_clusters)]
        result_y = [[] for _ in range(self.n_clusters)]
        for i in range(data.shape[0]):
            result_x[np.argmax(self.distribution[i,:])].append(data[i][0])
            result_y[np.argmax(self.distribution[i,:])].append(data[i][1])
            # result[np.argmax(self.distribution[i,:])].append(data[i])

        return result_x,result_y

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    '''
    generate data points, shape(number, 2)
    '''
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
    # plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    # plt.scatter(X1[:, 0], X1[:, 1], s=5)
    # plt.scatter(X2[:, 0], X2[:, 1], s=5)
    # plt.scatter(X3[:, 0], X3[:, 1], s=5)
    # plt.show()
    # import ipdb;ipdb.set_trace()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    res_x,res_y = gmm.predict(X)

    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    # import ipdb;ipdb.set_trace()

    plt.scatter(res_x[0], res_y[0], s=5)
    plt.scatter(res_x[1], res_y[1], s=5)
    plt.scatter(res_x[2], res_y[2], s=5)
    # plt.scatter(X2[:, 0], X2[:, 1], s=5)
    # plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    # import ipdb;ipdb.set_trace()
    print(res_x,res_y)
    # print(cat)
    # 初始化

    

