#-*- coding:utf-8 -*-
from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    #将一个矩阵的每一行向量相加 
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #返回distances由大到小的值对应的下标 
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), #python2  classCount.iteritems()
    key=operator.itemgetter(1),reverse=True)
    #取下标为1的元素作为排序权重，排序后翻转（因为默认的排序结果是是由小到大） 
    return sortedClassCount[0][0]
