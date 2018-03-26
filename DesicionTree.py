# -*- coding: utf-8 -*-
# Decision-tree.py
import numpy as np
from functools import *
import pydotplus
import re

feature = {}


def predict_by_tree(sample, tree):
    pass


def counts_by_feature_and_class(data, ft, ft_val, cls=None):
    #total是个[True False]矩阵，表示满足条件的行是否存在
    if cls == None:
        total = data[:, ft] == ft_val
    else:
        # logical_and可以基于多个列条件筛选，得到新的[True False]，作为数据统计
        total = np.logical_and(data[:,ft]==ft_val,\
        data[:,(data.shape[1]-1)]==cls )
    ret = np.sum(total == True)
    # print('ft=',ft,'ft_val=',ft_val,'cls=',cls,'counts',ret)
    #不能算log2(0),用极小的值代替0
    if ret == 0: ret = 0.00001
    return ret


#skip_features 表示本次计算gDA需要跳过的特征
#划分的越深，特征集逐渐减少
def counts_gDA(data_set, skip_features=None, method='id3'):
    #print(data_set.shape)
    sample_nums, feature_nums = data_set.shape
    feature_nums -= 1
    # print(sample_nums,feature_nums)
    #print(data_set[:,feature["class"]])
    class_vars, class_counts = np.unique(
        data_set[:, (data_set.shape[1] - 1)], return_counts=True)
    # print(class_vars,class_counts)

    class_p = np.array(class_counts) / sample_nums
    #print(class_p)
    #计算样本信息熵H_D
    H_D = -1 * np.sum(class_p * np.log2(class_p))
    print('H_D:', H_D)

    class_vars, class_counts = np.unique(
        data_set[:, (data_set.shape[1] - 1)], return_counts=True)

    # feature_vars矩阵 的每一行即是对应每个特征的分量
    # feature_counts矩阵 的每一行即是对应每个特征在样本中出现的次数
    feature_vars = []
    feature_counts = []
    for i in range(feature_nums):
        if skip_features != None and i in skip_features:
            print('skip feature:', i)
            continue
        var, counts = np.unique(data_set[:, i], return_counts=True)
        feature_vars.append(var)
        feature_counts.append(counts)

    #对每个特征，求条件熵
    # p(Y|X) X是条件，Y是结果
    #找到某列等于a 并且最后一列等于0的个数
    #特征0，特征分量为0,并且y=0的数目
    #找出refuge allow refuge_refuse的所有样本
    # refuge_allow =data_set[data_set[:,feature["class"]]==name["refuge"]["allow"]]
    # print(refuge_allow)

    # H_Y_x 表示某个特征(X)的某个值(x)下的条件熵，计算所有的x后累加得到最终的
    H_Y_X = []
    for i in range(len(feature_vars)):
        # H_Y_X的最终关于条件X的条件熵
        H_Y_x = []
        for j in feature_vars[i]:
            #对每一个特征的每一个特征分量,计算
            #p(y=allow|feature=a) + p(y=refuse|feature=a)
            #前半p(y=allow,feature=a) / p(feature = a)
            #先统计p(y=allow,feature=a)
            #如何统计矩阵中，列feature==a,y==alow的行数?要足够完美
            #tmp = a[a[:,0]==0,:]
            #res =tmp[tmp[:,1]==1,:] 还不够完美
            #下面的完美:
            # map_reduce..
            tmp = map(lambda x:counts_by_feature_and_class(data_set,i,j,x)/counts_by_feature_and_class(data_set,i,j),\
            class_vars)
            tmp = np.array(list(tmp))
            # print('tmp',tmp)
            tmp = np.log2(tmp) * tmp
            # print('tmp log',tmp)
            p_x = counts_by_feature_and_class(data_set, i, j)
            H_Y_x.append(np.sum(tmp) * p_x / sample_nums)
        H_Y_X.append(-sum(H_Y_x))
    # print('H_Y_X for feature:',H_Y_X)
    H_Y_X = np.array(H_Y_X)

    #计算信息增量 g(D,a) = H(D) - H(D|a)
    if (method == 'id3'):
        g_D_A = H_D - H_Y_X
    elif (method == 'c4.5'):
        g_D_A = (H_D - H_Y_X) / H_D
    else:
        g_D_A = H_D - H_Y_X

    return np.array(class_vars), feature_vars, np.array(g_D_A)


#g_D_A 按最大值取特征,作为划分
# 可能为interal node(0) or leaf node(1)
# interal node 记录特征
# leaf node 记录分类结果中的一个

full_fts = []
node_lists = []
skip_features = []


#每次迭代都应该引起特征集合的减少
def create_node(data, *, father=None, feature_val=[0], method='id3'):
    global full_fts, skip_features
    # 输入是按某特征的特征空间划分的数据子集，
    if skip_features == []:
        loop = [data]
    else:
        loop = data
    for j, d in enumerate(loop):
        if d == []:
            print('data sets null')
            return
        #into np.array
        # d = np.array(d)
        # print('d',d)
        print('子集: ', d.shape)
        # print('特征值空间(划分子集的依据)', feature_val)
        print('j=', j)
        if d.shape[0] == 0:
            print('data sets for feature %s null', feature_val[j])
            continue

        # 如果d已经全部属于某一类，则创建相应节点
        vals, counts = np.unique(d[:, d.shape[1] - 1], return_counts=True)
        # print('dataset vals',vals)
        # print('dataset counts',counts)
        if len(vals) == 1 and len(counts) == 1 and counts[0] == d.shape[0]:
            print('create leaf: all samples belongs to class:', vals[0])
            dt_node(
                idx=-1,
                property='leaf',
                father=father,
                val=vals[0],
                ft_val=feature_val[j])
            continue
        #skip_feature满了,说明特征集用完了，树即将构建完成,选择剩余数据中较多的那个类作为叶节点
        if full_fts != []:
            if (sum(skip_features) == sum(list(range(len(full_fts))))):
                # or len(feature_val) >= d.shape[0] \
                #特征值空间 [0, 1, 2],子集:  (2, 5)
                #如果出现特征值空间还大于子集合个数了，有点异常,同样按谁数目多，就按谁来取
                if len(feature_val) > d.shape[0]:
                    print('val in feature sets more than sample! weird')

                vals, counts = np.unique(
                    d[:, (d.shape[1] - 1)], return_counts=True)
                print('vals=', vals)
                print('counts=', counts)
                vals = vals.tolist()
                counts = counts.tolist()
                print('counts type', type(counts))
                idx = counts.index((max(counts)))
                print('create leaf: feature_set empty!choose max', vals, idx,
                      vals[idx])
                dt_node(
                    idx=-1,
                    property='leaf',
                    father=father,
                    val=vals[idx],
                    ft_val=feature_val[j])
                continue
        #分类 特征集 特征对应gda
        cs, fts, g_D_A = counts_gDA(d, skip_features, method=method)
        print('分类:',cs)
        print('feature sets:', fts)
        print('method=', method)
        if method == 'id3':
            print('信息增量 g_D_A:', g_D_A)
        if method == 'c4.5':
            print('信息增量比 r_D_A:', g_D_A)
        # 如果skip_feature为空，说明是第一次迭代，记录初始的分类和特征
        if skip_features == []:
            full_fts = fts
        #找到最大gda的特征
        gDA = g_D_A.tolist()
        #尝试下，为了idx是对齐的 在gDA里补充skip_feature的列为-1!
        for v in skip_features:
            gDA.insert(v, -1)
        print('gDA final list=', gDA)
        if max(gDA) < 0.1:
            vals, counts = np.unique(
                d[:, (d.shape[1] - 1)], return_counts=True)
            vals = vals.tolist()
            counts = counts.tolist()
            idx = counts.index((max(counts)))
            print('create leaf: gDA too low!choose max', vals, idx, vals[idx])
            dt_node(
                idx=-1,
                property='leaf',
                father=father,
                val=vals[idx],
                ft_val=feature_val[j])
            continue
        idx = gDA.index(max(gDA))
        #按照该特征找到对应的特征集,并划分子数据集
        # print('spited by features:',fts[idx])
        # skip_features = []
        #在递归子集里,idx=2和上一次同样为idx为2表示的特征已经不一样了
        #需要解算出在原始fts里的真正的index来skip
        # 如果idx >= skip_features的最后一个元素 则+1
        real_idx = idx
        # if skip_features != []:
        #     if idx >= skip_features[-1]:
        #         real_idx  = idx +1

        print('idx=', idx, 'real_idx=', real_idx)

        #children表示依据real_idx描述的特征来划分数据集
        #应该把对应的特征的值也保存在child里
        #i对某个应特征集合中的具体值
        children = []
        child_feature_val = []
        for i in full_fts[real_idx]:
            children.append(d[d[:, real_idx] == i, :])
            child_feature_val.append(i)
        # print('children',children)
        print('before append sk:', skip_features)
        if real_idx not in skip_features:
            skip_features.append(real_idx)
        print('after append sk:', skip_features)
        print('-----create internal node------')
        node = dt_node(
            idx=real_idx,
            property='internal',
            father=father,
            val=-1,
            ft_val=feature_val[j])

        #采用skip的思路,而不是先计算减少的特征集，可以避免数据块的搬移，速度要更快
        create_node(
            children,
            father=node,
            feature_val=child_feature_val,
            method=method)

        #feature一旦跳出循坏，意味着递归回退，最近添加的skip_feature的feature又要需要删除保证递归正确
    if skip_features != []:
        skip_features.remove(skip_features[-1])


last_name = []
counts = 0


class dt_node(object):
    def __init__(self, **kwds):
        global counts, last_name
        self.node = kwds
        # print('new node',self.node)

        # 根据value 反查key
        rever_dict = {v: k for k, v in feature.items()}
        name = rever_dict[kwds['idx']]
        if name in last_name:
            name = name + str(counts)
            counts += 1
        self.name = name
        last_name.append(name)

        node_lists.append(self)
        if self.node['father'] == 'root':
            print(self.name, '---father-->:', self.node['father'])
        else:
            print(self.name, '---father-->:', self.node['father'].name)
        # print('node:', self.node)


#可视化决策树
# sudo apt-get install graphviz
# conda install pydotplus
def visiualization(lists):
    g = pydotplus.Dot(graph_type='digraph')
    i = 0
    for item in lists:
        #不同node强制用不同的名字
        node_name = item.name
        if 'leaf' in node_name and item.node['val'] != -1:
            node_name = "leaf" + str(i) + "\n" + str(item.node['val'])
            i += 1
        node = pydotplus.Node(node_name)
        if item.node['property'] == 'internal':
            node.set('shape', 'diamond')
        else:
            node.set('shape', 'box')

        father_node_name = item.node['father']
        if father_node_name != 'root':
            father_node_name = item.node['father'].name
        father_node = pydotplus.Node(father_node_name)

        edge = pydotplus.Edge(father_node_name, node_name)
        if 'ft_val' in list(item.node.keys()):
            edge.set('label', item.node['ft_val'])
        g.add_node(node)
        g.add_node(father_node)
        g.add_edge(edge)

    g.write("dt-tree.png", format='png')
    import os
    os.system("eog dt-tree.png")


import csv




#将数据集数值化成为如下矩阵
# age 0-青年 1-中年 2-老年
# job 1-有 0-无
# house 1-有 0-无
# bank  1-一般 2-好 3-非常好
# class 1-允许贷款 0-不允许
# |age |job |house  |bank   |class |
# | 0  | 0  |  0    |  1    | 0    |
def book_demo():
    demo_data = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0],
        [0, 1, 0, 2, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 2, 0],
        [1, 1, 1, 2, 1],
        [1, 0, 1, 3, 1],
        [1, 0, 1, 3, 1],
        [2, 0, 1, 3, 1],
        [2, 0, 1, 2, 1],
        [2, 1, 0, 2, 1],
        [2, 1, 0, 3, 1],
        [2, 0, 0, 1, 0],
        # new added
    ]
    feature["leaf"] = -1
    feature["age"] = 0
    feature["job"] = 1
    feature["house"] = 2
    feature["bank"] = 3
    feature["class"] = 4
    create_node(np.array(demo_data), father='root', method='id3')
    visiualization(node_lists)
    pass


def kaggle_Tatanic_data():

    def processing_data(trainfile):
        with open(trainfile, "r", encoding="utf-8") as f:
            rd = csv.reader(f)
            rows = [row for row in rd]
        return rows

    data = processing_data('/opt/study/python3/AI/train.csv')
    #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    #去掉id,name,Ticket,Fare,Carbin
    data = np.array(data)
    data = np.delete(data, [0, 3, 5,8, 9, 10], axis=1)
    #新的年龄在第几列?
    # row, col = np.where(data == 'Age')
    #字符float化,还有''未知年龄...怎么处理
    # tmp = [np.float(x) for x in data[1:, col[0]]]
    # tmp = np.array(tmp)
    # data[int(data[:, col[0]]) <= 8, col[0]] = u'young'
    # data[data[:, col[0]] <= 60 and data[:, col[0]] >= 8, col[0]] = 60
    # data[data[:, col[0]] >= 60, col[0]] = 80

    #把此时的第0列挪到最后一列
    data = data[:, ::-1]
    #去掉第一行
    data = np.delete(data, [0], axis=0)

    feature["leaf"] = -1
    feature["embark"] = 0
    feature["parch"] = 1
    feature["sibsp"] = 2
    feature["sex"] = 3
    feature["pclass"] = 4
    feature["class"] = 5

    create_node(np.array(data), father='root', method='id3')
    visiualization(node_lists)
    pass


import pandas as pd


def kaggle_Tatanic_demo_better():
    train = pd.read_csv('/opt/study/python3/AI/train.csv')
    test = pd.read_csv('/opt/study/python3/AI/test.csv')

    # train.head(3)
    # Copy original dataset in case we need it later when digging into interesting features


    # WARNING: Beware of actually copying the dataframe instead of just referencing it
    # "original_train = train" will create a reference to the train variable (changes in 'train' will apply to 'original_train')
    original_train = train.copy(
    )  # Using 'copy()' allows to clone the dataset, creating a different object with the same values

    # Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings
    full_data = [train, test]

    # Feature that tells whether a passenger had a cabin on the Titanic
    train['Has_Cabin'] = train["Cabin"].apply(
        lambda x: 0 if type(x) == float else 1)
    test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Create new feature FamilySize as a combination of SibSp and Parch
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # Create new feature IsAlone from FamilySize
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # Remove all NULLS in the Embarked column
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    # Remove all NULLS in the Fare column
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    # Remove all NULLS in the Age column
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(
            age_avg - age_std, age_avg + age_std, size=age_null_count)
        # Next line has been improved to avoid warning
        dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)


    # Define function to extract titles from passenger names
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""


    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace([
            'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
            'Jonkheer', 'Dona'
        ], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping titles
        title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({
            'S': 0,
            'C': 1,
            'Q': 2
        }).astype(int)

        # Mapping Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454),
                    'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31),
                    'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age']

        # Feature selection: remove variables no longer containing relevant information
        drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
        train = train.drop(drop_elements, axis = 1)
        test  = test.drop(drop_elements, axis = 1)

        # move 'Surivied' to the last column
        surived =  train.pop('Survived')
        train.insert(0,'Survided',surived)

        drop_elements = ['FamilySize','Parch']
        train = train.drop(drop_elements, axis = 1)
        test  = test.drop(drop_elements, axis = 1)

        train_np = np.array(train)
        test_np = np.array(test)
        print(train_np)

        feature["leaf"] = -1
        feature["Age"] = 0
        feature["Embarked"] = 1
        # feature["FamilySize"] = 2
        feature["Fare"] = 2
        feature["Has_Carbin"] = 3
        feature["IsAlone"] = 4
        # feature["Parch"] = 6
        feature["Pclass"] = 5
        feature["Sex"] = 6
        feature["class"] =7

        train.head(0)
        create_node(train_np, father='root', method='id3')
        visiualization(node_lists)
        


        pass

if __name__ == '__main__':

    # book_demo()
    kaggle_Tatanic_data()
    # kaggle_Tatanic_demo_better()

