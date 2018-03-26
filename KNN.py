# -*- coding: utf-8 -*-
# KNN.py



import numpy as np
import matplotlib.pyplot as plt
import cv2



if __name__ == '__main__':

    # 记住numpy的array有下面这种神器的用法
    test = np.array([1,-2,3,-4,5])
    res = test[test > 0]

    # res = [1,3,5]
    a = np.random.randint(0,101,(50,2),dtype=np.int64).astype(np.float32)
    flag = np.random.randint(0,2,(50,1),dtype = np.int64).astype(np.float32)
    red =  a[flag.ravel() == 0]
    blue =  a[flag.ravel() == 1]
    plt.figure
    plt.scatter(red[:,0],red[:,1],40,'r')
    plt.scatter(blue[:,0],blue[:,1],40,'b','<')

    newcomer = np.random.randint(0,101,(1,2),dtype=np.int64).astype(np.float32)

    # draw a cycle
    r = 20
    theta = np.arange(0, 2*np.pi, 0.01)
    x = newcomer[0,0] + r * np.cos(theta)
    y = newcomer[0,1] + r * np.sin(theta)
    plt.plot(x, y,'r-')
    plt.axis('tight')


    plt.scatter(newcomer[:,0],newcomer[:,1],40,'g','>')

    knn = cv2.ml.KNearest_create()
    knn.train(a,cv2.ml.ROW_SAMPLE,flag)

    ret,results,neighbours,dist = knn.findNearest(newcomer,5)

    print('ret= ',ret)
    print('results=',results)
    print('neighbours=',neighbours)
    print('dist=',dist)

    plt.show()


