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
    p = np.sum(alpha[:,i]);
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
    p = np.sum(Pi * B[:,obs[i]] * beta[:,i]);

    return p,beta;

# 可以计算观测预测时刻t,某状态为qi的概率;
# 可以计算观测预测时刻t,某状态为qi,时刻t+1,某状态为qj的联合概率;
# 根据观测预测某个状态在T内出现的期望值..
def p_cal(Pi,A,B,obs):

    times = len(obs);
    [n,m] = B.shape;

    [p1,alpha] = forward_method(Pi,A,B,obs);
    [p2,beta]  = backward_method(Pi,A,B,obs);

    #666
    p3 = np.sum (np.dot(A , B[:,obs[-1]] * beta[:,-1]) * alpha[:,-2]);

    # 观测预测时刻t,某状态为qi的概率; gamma
    gamma = np.zeros([n,times]);
    gamma = alpha * beta;  
    s = np.sum(gamma,axis=0);
    gamma = gamma / s[None,:];

    #在已经观测下,计算预测时刻t,某状态为qi,时刻t+1,某状态为qj的联合概率;
    # 实际上最后1次没有算，仅算到1到T-1
    xi = np.zeros([n,n,times]);
    parts_martix = np.zeros([n,n]);

    for t in range(times-1):
        tmp = np.dot(alpha[:,t][:,None],(beta[:,t+1]*B[:,obs[t+1]])[None,:]);
        parts_martix =  tmp * A; 
        xi[:,:,t] =  parts_martix / np.sum(parts_martix);


    return p3,alpha,beta,gamma,xi

#输入观测序列，初始矩阵，迭代次数
#输出估计参数
def baum_welch_estimate(obs,init_Pi,init_A,init_B,iter_times):


    times = len(obs);
    [n,m] = init_B.shape;
    #计算b的核心是构造I矩阵，
    #k=0,t时刻的观测值必须等于vk才有效,
    #观测序列为obs[0 1 0]
    # k=0 时，得到 [ 1 0 1], 1 代表有效，代表无效
    # k=1 时，得到[0 1 0]
    #用数学来表达，构造一个矩阵
    #       0  0 0 0 0
    # tmp = 1  1 1 1 1  obs = [2 1 0 1] 
    #       2  2 2 2 2 
    #tmp的每一行与obs求与即可
    #和b的维度要一致，
    #肯定是个nxt x txm, nxt是gamma的维度，所以构造的矩阵为txm
    #这里直接构造好了
    # mxt
    I = np.ones([m,times]);
    for i in range(m):
        I[i,:] = I[i,:] * i;

    I = np.where(I==obs,1,0) ;
    #txm
    I = I.T;
    


    A = np.zeros(init_A.shape);
    B = np.zeros(init_B.shape);
    Pi = np.zeros(init_Pi.shape);

    for t in range(iter_times):
        print('iteration ',t+1,'..')
        if(t == 0):
            p,alpha,beta,gamma,xi = p_cal(init_Pi,init_A,init_B,obs);
        else:
            p,alpha,beta,gamma,xi = p_cal(Pi,A,B,obs);
        #迭代计算A,B,Pi

        # 因为最后一个xi[:,:,t]为全0，
        # 但是gamma不是,计算时要注意
        A = np.sum(xi[:,:,:-1],axis=2) / np.sum(gamma[:,:-1],axis=1)[:,None];

        
        B = np.dot(gamma,I) / np.sum(gamma,axis=1)[:,None] ;

        Pi = gamma[:,0];


    return A,B,Pi



from hmmlearn import hmm;
def hmm_learn_api(A,B,Pi):
    states = ["box 1", "box 2", "box3"]
    n_states = len(states)
    observations = ["red", "white"]
    n_observations = len(observations)
    model2 = hmm.MultinomialHMM(n_components=n_states,
    startprob_prior=Pi,transmat_prior=A, n_iter=200, tol=0.01)
    # X2 = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1]])
    X2 = np.array([[0,1,0]])
    model2.fit(X2)
    print('PI',model2.startprob_)
    print('A',model2.transmat_)
    print('B',model2.emissionprob_)
    print('score',model2.score(X2))
    # model2.fit(X2)
    # print (model2.startprob_)
    # print (model2.transmat_)
    # print (model2.emissionprob_)
    # print (model2.score(X2))
    # model2.fit(X2)
    # print (model2.startprob_)
    # print (model2.transmat_)
    # print (model2.emissionprob_)
    # print (model2.score(X2))





if __name__ == '__main__':
    #   | 0.5 0.3 0.2 |    |0.5 0.5|
    # A=| 0.3 0.5 0.2 | B= |0.4 0.6|
    #   | 0.2 0.3 0.5 |    |0.7 0.3|
    obs = [0,1,0];

    A=[[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]];
    B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]];
    Pi=[0.2,0.4,0.4];

    A =np.array(A);
    B=np.array(B);
    Pi =np.array(Pi);
    obs =np.array(obs);

    print('observe seqence probability cal(fp):',forward_method(Pi,A,B,obs));
    print('observe seqence probability cal(bp):',backward_method(Pi,A,B,obs));
    print('observe seqence probability cal(total):',p_cal(Pi,A,B,obs));
    print('veterbi cal:',veterbi_cal(Pi,A,B,obs));

    # init_A = np.ones(A.shape) * 0.33333;
    # init_B = np.ones(B.shape) * 0.5;
    # init_Pi = np.ones(Pi.shape) * 0.3333;

    A=[[0.33,0.32,0.35],[0.32,0.33,0.35],[0.35,0.31,0.34]];
    B=[[0.6,0.4],[0.3,0.7],[0.7,0.3]];
    Pi=[0.33,0.34,0.33];

    init_A =np.array(A)
    init_B=np.array(B)
    init_Pi =np.array(Pi)

    A,B,Pi = baum_welch_estimate(obs,init_Pi,init_A,init_B,2);
    print('model estimate A\n',A);
    print('model estimate B\n',B);
    print('model estimate Pi\n',Pi);

    print('hmmlearn api');
    hmm_learn_api(init_A,init_B,init_Pi);

