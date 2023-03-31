# decision_tree.py 决策树

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 信息熵
def ent(data):
    datalabel = data.iloc[:, -1]
    labelclass = datalabel.value_counts()  # 总共有多少类
    s = 0
    for i in labelclass.keys():
        pk = labelclass[i] / len(datalabel)
        s += -pk * np.log2(pk)
    return s


# 离散信息增益
def information_gain(data, a):
        Ent = ent(data)
        featureclass = data[a].value_counts()  # 特征有多少种可能
        gain = 0
        for v in featureclass.keys():
            weight = featureclass[v] / data.shape[0]
            Ent_v = ent(data.loc[data[a] == v])
            gain += weight * Ent_v
        return Ent - gain


# 连续信息增益
def information_gain_continuous(data, a):
    n = len(data)
    data_a_value = sorted(data[a].values)  # 排序
    Ent = ent(data)  # 原始数据集的信息熵
    select_points = []
    for i in range(n - 1):
        val = (data_a_value[i] + data_a_value[i + 1]) / 2  # 两个值中间取值为划分点
        data_left = data.loc[data[a] < val]
        data_right = data.loc[data[a] > val]
        ent_left = ent(data_left)
        ent_right = ent(data_right)
        result = Ent - len(data_left) / n * ent_left - len(data_right) / n * ent_right
        select_points.append([val, result])
    select_points.sort(key=lambda x: x[1], reverse=True)  # 按照信息增益排序
    return select_points[0][0], select_points[0][1]  # 返回信息增益最大的点, 以及对应的信息增益


# 获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:, -1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]


# 获取最佳划分特征
def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        if a in continuous_features:
            temp_val, temp = information_gain_continuous(data, a)
            res[a] = [temp_val, temp]
        else:
            temp = information_gain(data, a)
            res[a] = [-1, temp]  # 离散值没有划分点，用-1代替
    res = sorted(res.items(), key=lambda x: x[1][1], reverse=True)
    return res[0][0], res[0][1][0]


# 将数据转化为（属性值：数据）的元组形式返回，并删除之前的特征列，只针对离散数据
def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data


# 创建决策树
def create_tree(data):
    data_label = data.iloc[:, -1]
    if len(data_label.value_counts()) == 1:  # 只有一类
        return data_label.values[0]
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:, :-1].columns):  # 所有数据的特征值一样，选样本最多的类作为分类结果
        return get_most_label(data)
    best_feature, best_feature_val = get_best_feature(data)  # 根据信息增益得到的最优划分特征
    if best_feature in continuous_features:  # 连续值
        node_name = best_feature + '<' + str(best_feature_val)
        Tree = {node_name: {}}  # 用字典形式存储决策树
        Tree[node_name]['是'] = create_tree(data.loc[data[best_feature] < best_feature_val])
        Tree[node_name]['否'] = create_tree(data.loc[data[best_feature] > best_feature_val])
    else:
        Tree = {best_feature: {}}
        exist_vals = pd.unique(data[best_feature])  # 当前数据下最佳特征的取值
        if len(exist_vals) != len(column_count[best_feature]):  # 如果特征的取值相比于原来的少了
            no_exist_attr = set(column_count[best_feature]) - set(exist_vals)  # 少的那些特征
            for no_feat in no_exist_attr:
                Tree[best_feature][no_feat] = get_most_label(data)  # 缺失的特征分类为当前类别最多的
        for item in drop_exist_feature(data, best_feature):  # 根据特征值的不同递归创建决策树
            Tree[best_feature][item[0]] = create_tree(item[1])
    return Tree


# 根据创建的决策树进行分类
def predict(Tree, test_data):
    first_feature = list(Tree.keys())[0]
    if (feature_name := first_feature.split('<')[0]) in continuous_features:
        second_dict = Tree[first_feature]
        val = float(first_feature.split('<')[-1])
        input_first = test_data.get(feature_name)
        if input_first < val:
            input_value = second_dict['是']
        else:
            input_value = second_dict['否']
    else:
        second_dict = Tree[first_feature]
        input_first = test_data.get(first_feature)
        input_value = second_dict[input_first]
    if isinstance(input_value, dict):  # 判断分支还是不是字典
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label


# 绘制带箭头的注解
def plotnode(nodetext, centerpt, parentpt, nodetype):
    createplot.ax1.annotate(nodetext, xy=parentpt, xycoords='axes fraction',
                            xytext=centerpt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodetype, arrowprops=arrow)


def createplot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createplot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plottree.totalW = float(getleafnum(intree))
    plottree.totalD = float(gettreedepth(intree))
    plottree.xOff = -0.5/plottree.totalW
    plottree.yOff = 1.0
    plottree(intree, (0.5, 1.0), '')
    plt.show()


# 获取叶结点数量
def getleafnum(tree):
    num = 0
    firststr = list(tree.keys())[0]
    seconddict = tree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            num += getleafnum(seconddict[key])
        else:
            num += 1
    return num


# 数的层数
def gettreedepth(tree):
    num = 0
    firststr = list(tree.keys())[0]
    seconddict = tree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':
            thisdepth = 1 + gettreedepth(seconddict[key])
        else:
            thisdepth = 1
        if thisdepth > num: num = thisdepth
    return num


# 父子结点间文本
def plotmidtext(cntrpt, parentpt, textstring):
    xmid = (parentpt[0]-cntrpt[0])/2.0+cntrpt[0]
    ymid = (parentpt[1]-cntrpt[1])/2.0+cntrpt[1]
    createplot.ax1.text(xmid, ymid, textstring)


def plottree(tree, parentpt, nodetxt):
    # 计算宽高
    numleafs = getleafnum(tree)
    depth = gettreedepth(tree)
    firststr = list(tree.keys())[0]
    cntrpt = (plottree.xOff + (1.0 + float(numleafs))/2.0/plottree.totalW, plottree.yOff)
    plotmidtext(cntrpt, parentpt, nodetxt)
    # 标记子节点属性
    plotnode(firststr, cntrpt, parentpt, decisionnode)
    seconddict = tree[firststr]
    # 减少y偏移
    plottree.yOff = plottree.yOff - 1.0/plottree.totalD
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            plottree(seconddict[key], cntrpt, str(key))
        else:
            plottree.xOff = plottree.xOff + 1.0/plottree.totalW
            plotnode(seconddict[key], (plottree.xOff, plottree.yOff), cntrpt, leafnode)
            plotmidtext((plottree.xOff, plottree.yOff), cntrpt, str(key))
    plottree.yOff = plottree.yOff + 1.0/plottree.totalD


if __name__ == '__main__':
    data = pd.read_csv('西瓜数据集3.0.csv')
    # 统计每个特征的取值情况作为全局变量
    column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])
    test = information_gain_continuous(data, '密度')
    continuous_features = ['密度', '含糖率']  # 先标注连续值
    decide_tree = create_tree(data)
    plt.rc('font', family='SimHei')  # 设置字体为黑体
    plt.rc('axes', unicode_minus=False)  # 解决坐标轴负号显示问题
    # 文本框与箭头格式
    decisionnode = dict(boxstyle='sawtooth', fc="0.8")
    leafnode = dict(boxstyle="round4", fc="0.8")
    arrow = dict(arrowstyle="<-")
    createplot(decide_tree)
