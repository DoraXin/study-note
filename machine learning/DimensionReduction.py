import numpy as np

def pca(X,rate = 1):
    '''
    对数据进行PCA降维.

    参数:
        X:shape(n_features,m_samples),每一列代表一个样本.
        rate:期望保留多少方差,default = 100%
    -----------
    输出:
        p:投影矩阵
        mu:均值向量
    '''
    rows,column = X.shape
    def centeralise(X):
        '''
        将数据进行中心化处理.
        -----------------
        输入数据:
            X:每一列代表一个样本.
        ---------
        输出数据:
            X_new:进行中心化处理后的样本矩阵.
            mu:均值向量
        '''
        mu = (X.sum(axis=1))/column #将每一列加到第一列
        # sigam = 
        for i in range(column):
            X[:,i] =  X[:,i] - mu
        return X,mu

    X,mu = centeralise(X)
    u,sigma,v = np.linalg.svd(X.dot(X.T))
    total = sigma.sum()
    temp = 0
    while temp < sigma.size and sigma[:temp+1].sum()/total < rate:
        temp +=1
    p = v[:,0:temp+1] # 取出v的前temp列
    return p,mu

def kpca(X,k,rate = 1):
    '''
    对数据进行KPCA降维.

    参数:
        X:shape(n_features,m_samples),每一列代表一个样本.
        k:核函数
        rate:期望保留多少方差,default = 100%
    -----------
    输出:
        p:投影矩阵
        mu:均值向量
    '''
    rows,column = X.shape
    def centeralise(X):
        '''
        将数据进行中心化处理.
        -----------------
        输入数据:
            X:每一列代表一个样本.
        ---------
        输出数据:
            X_new:进行中心化处理后的样本矩阵.
            mu:均值向量
        '''
        mu = (X.sum(axis=1))/column #将每一列加到第一列
        for i in range(column):
            X[:,i] =  X[:,i] - mu
        return X,mu

    X,mu = centeralise(X)
    u,sigma,v = np.linalg.svd(k(X.T).dot(k(X))
    total = sigma.sum()
    temp = 0
    while temp < sigma.size and sigma[:temp+1].sum()/total < rate:
        temp +=1
    p = v[:,0:temp+1] # 取出v的前temp列
    return p,mu
  
                              
