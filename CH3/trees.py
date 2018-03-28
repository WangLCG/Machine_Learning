#-*- coding:utf-8 -*-
from math import log
import operator

#计算香农熵 
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#书中表3.1的数据集
def createDataSet():
    dataSet = [[1, 1,'yes'],[1, 1,'yes'],[1, 0,'no'],[0, 1,'no'],[0, 1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels
"""
    功能： 分类，从dataSet矩阵的第axis列筛选出值为value的元素所在的行
    dataSet:待分类数据，矩阵，矩阵每一行代表一个样本，每一列代表一个特征
    axis:  dataSet矩阵的第axis行，即第axis个特征
    value: 需要筛选出来的第axis特征的值
    返回：   筛选出的元素所在行组成的新矩阵（已经去除掉第axis列的值）
"""
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

"""
    功能：选择最优特征进行筛徐分类
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]  #获取第i个特征所有的可能取值  
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if(infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
    return bestFeature

"""
    功能:对使用完所有属性后依旧没有分类结果的数据进行投票，投票结果为该数据的最终的标签
    classList:已有的分类标签
    return:投票的结果，标签
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), #python2  classCount.iteritems()
                                  key=operator.itemgetter(1),reverse=True)
    #取下标为1的元素作为排序权重，排序后翻转（因为默认的排序结果是是由小到大）  
    return sortedClassCount[0][0]

#tree-building 
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] #所有类型一致时返回
    if len(dataSet[0]) == 1:
        return majorityCnt(classList) #当dataSet只有一个样本时，取出现最多的类标签作为其最终结果
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeateLabel = labels[bestFeat]
    myTree = {bestFeateLabel:{}} #用字典存储决策树 
    del(labels[bestFeat])   #使用完的特征从特征数组中删去 
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  #将labels拷贝到subLabels这个列表中 
        myTree[bestFeateLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree






