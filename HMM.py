# -*- coding: utf-8 -*-
# HMM.py


import numpy as np

#   | 0.5 0.3 0.2 |    |0.5 0.5|
# A=| 0.3 0.5 0.2 | B= |0.4 0.6|
#   | 0.2 0.3 0.5 |    |0.7 0.3|
A=[[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]];
B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]];
Pi=[0.2,0.4,0.4];
obs = [0,1,0];


# delta nx1
# A nxn
# return nx1
def max_delta(delta,A):
    [m,n]=A.shape;
    vec=np.zeros([m,1]);
    idx=np.zeros([m,1]);
    for i in range(n):
        vec[i] = np.max(A[:,i] * delta);
        idx[i] = np.argmax(A[:,i] * delta);
    return vec.reshape(1,-1),idx.reshape(1,-1);


# 状态空间 [1 2 3] 观测空间 [1，2](红 白)
# 预测序列 p={红，白，红}
# 维特比算法预测状态序列:
def veterbi_cal(Pi,A,B,obs):
    seq = obs;
    times = len(seq);
    A =np.array(A);
    B=np.array(B);
    Pi =np.array(Pi);
    obs =np.array(obs);
    #n为状态空间大小，m为观测空间大小
    [n,m]=B.shape;
    # delta初始化, 每1列代表1次观测,一共
    # delta = [d(0),d(1),....d(times-1)]
    # vaphi 类似
    delta = np.zeros([n,times]);
    vaphi = np.zeros([n,times],dtype=np.int64);

    for i in range(len(seq)):
        if(i == 0):
            delta[:,i] = Pi * B[:,obs[i]];  
            vaphi[:,i] = vaphi[:,i];
        else:
            # (1xn).T * (nx1) = nx1               
            # 然后扩充到times列 nxtimes 
            vec,idx  = max_delta(delta[:,i-1],A);
            delta[:,i] = vec * B[:,obs[i]]; 
            vaphi[:,i] = idx;
    
    state = [];
    val = np.max(delta[:,i]);
    # index是最后一个状态值的index
    index = np.argmax(delta[:,i]);
    state.insert(0,index);
    # 反溯
    # vaphi[:i]是类似[0;1;2]的向量，i当前反溯的最优路径的节点,vi是上一最优路径节点
    for i in range(len(seq)-2,-1,-1):
        index = vaphi[:,i+1][int(index)];
        # 表头插入，表示反向
        state.insert(0,int(index));
    
    return state;



if __name__ == '__main__':

    print('veterbi cal:',veterbi_cal(Pi,A,B,obs));

