# -*- coding: utf-8 -*-
# Perception.py

import numpy as np

# samples should like ((3,3),(4,3),(1,1)),(1,1,-1)
def Perception(samples,judges):
    samples = np.array(samples)
    judges = np.array(judges)
    N = samples.shape[0]
    b = 0
    GramMartrix = np.zeros((3,3),dtype = np.int64)
    for row in range(N):
        for col in range(N):
            GramMartrix[row,col] = samples[row].dot(samples[col]) 
    # print(GramMartrix)
    alpha = np.zeros((1,N),dtype = np.int64)
    # k means times
    k = 0
    conts = 0
    ## for every sample, cal error
    ## last error and last last error all > 0 , exit loop
    while(conts < 3):
        col = k % N
        tmp =  GramMartrix[col,:]*judges*alpha
        # print('tmp = ',tmp)
        error = (tmp.sum() + b)*judges[col]
        # print('error = ',error)
        if error <= 0:
            alpha[0,col] = alpha[0,col] + 1
            b = b + judges[col]
            # print('col = ',col+1)
            # print('alpha = ',alpha)
            # print('b = ',b)
            conts = 0
        else:
            conts += 1
        k = k + 1
    # shape(1,2)
    tmp = [alpha[0,i]*samples[i]*judges[i] for i in range(N)]
    omega = sum(tmp)
    return omega,np.array(b)
    

if __name__ == '__main__':
    w,b = Perception(((3,3),(4,3),(1,1)),(1,1,-1))
    print('w = ',w,',b = ',b)
    

