# -*- coding: utf-8 -*-
# HMM.py

import numpy as np

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




# 计算观测obs的概率
# 有前向和后向两种实现方法,还可以综合使用，得出的结果都应该是一致的
# 输入obs = [0 1 0] (红白红)
def forward_method(Pi,A,B,obs):
    times = len(obs);
    #n为状态空间大小，m为观测空间大小
    [n,m]=B.shape;
    #alpha 初始化 alpha(i)全部保存下来
    alpha = np.zeros([n,times]);
    for i in range(times):
        if (i == 0):
            alpha[:,i] = Pi * B[:,obs[i]];
        else:
            alpha[:,i] = np.dot(A.T,alpha[:,i-1]) * B[:,obs[i]];
    #final
    p = sum(alpha[:,i]);
    return p,alpha;

def backward_method(Pi,A,B,obs):
    times = len(obs);
    #n为状态空间大小，m为观测空间大小
    [n,m]=B.shape;
    #后向倒序
    beta = np.ones([n,times]);
    for i in range(times-1,-1,-1):
        if(i == times -1):
            beta[:,i] = beta[:,i]; # keep 1
        else:
            beta[:,i] = np.dot(A,beta[:,i+1] * B[:,obs[i+1]]);
    #final
    p = sum(Pi * B[:,obs[i]] * beta[:,i]);

    return p,beta;

# 可以计算观测预测时刻t,某状态为qi的概率;
# 可以计算观测预测时刻t,某状态为qi,时刻t+1,某状态为qj的联合概率;
# 根据观测预测某个状态在T内出现的期望值..
def p_cal(Pi,A,B,obs):

    [p1,alpha] = forward_method(Pi,A,B,obs);
    [p2,beta]  = backward_method(Pi,A,B,obs);

    #666
    p3 = sum (np.dot(A , B[:,obs[-1]] * beta[:,-1]) * alpha[:,-2]);

    return p1,p2,p3

if __name__ == '__main__':
    #   | 0.5 0.3 0.2 |    |0.5 0.5|
    # A=| 0.3 0.5 0.2 | B= |0.4 0.6|
    #   | 0.2 0.3 0.5 |    |0.7 0.3|
    A=[[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]];
    B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]];
    Pi=[0.2,0.4,0.4];
    obs = [0,1,0];

    A =np.array(A);
    B=np.array(B);
    Pi =np.array(Pi);
    obs =np.array(obs);

    print('observe seqence probability cal(fp):',forward_method(Pi,A,B,obs));
    print('observe seqence probability cal(bp):',backward_method(Pi,A,B,obs));
    print('observe seqence probability cal(total):',p_cal(Pi,A,B,obs));
    # print('veterbi cal:',veterbi_cal(Pi,A,B,obs));

